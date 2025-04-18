"""
Core McpClient implementation with Gemini orchestration.
"""

import os
import sys
import json
import asyncio
import logging
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from contextlib import AsyncExitStack

from google import genai
from google.genai import types as genai_types
from google.api_core import exceptions as google_exceptions
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.syntax import Syntax
import questionary

from .config import Config, McpServerConfig, DEFAULT_MAX_HISTORY_TOKENS

logger = logging.getLogger(__name__)

# Task completion marker
TASK_COMPLETE_MARKER = "TASK_COMPLETE:"
MAX_AGENT_ITERATIONS = 7 # Limit consecutive tool call loops

class McpClient:
    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.gemini_model_name: str = config.get_default_model() # Load model name from config
        self.mcp_servers_config: Dict[str, McpServerConfig] = config.get_mcp_servers()
        self.max_tool_calls_per_turn: int = config.get_setting("max_tool_calls_per_turn", 5)
        self.confirm_risky_tools: bool = config.get_setting("confirm_risky_tools", True)
        self.max_history_tokens: int = config.get_setting("max_history_tokens", DEFAULT_MAX_HISTORY_TOKENS)
        self._tokenizer: Optional[tiktoken.Encoding] = None
        self.system_prompt = self._create_system_prompt()

        # --- Initialize History ---
        self.conversation_history: List[genai_types.Content] = [
            genai_types.Content(role="model", parts=[genai_types.Part(text="Okay, I'm ready. How can I help you?")]) # Initial model ack
        ]

        # --- Initialize Gemini Client ---
        # Moved initialization here AFTER history and prompt are ready
        self.gemini_client: Optional[genai.GenerativeModel] = None
        self._initialize_gemini()

        # --- Initialize Tokenizer ---
        self._initialize_tokenizer()

        logger.info("McpClient initialized.")

    def _initialize_gemini(self):
        """Initialize the Gemini client and model."""
        api_key = self.config.get_gemini_api_key()
        if not api_key:
            logger.error("Gemini API Key not found in config or environment.")
            self.console.print("[bold red]Error:[/bold red] Gemini API Key not configured. Please run `mcp-client setup-gemini <YOUR_API_KEY>` or set GEMINI_API_KEY environment variable.")
            raise ValueError("Gemini API Key not configured.")

        try:
            genai.configure(api_key=api_key)
            self.gemini_client = genai.GenerativeModel(
                model_name=self.gemini_model_name,
                system_instruction=self.system_prompt
                # GenerationConfig can be passed per-request if needed
            )
            # Quick check if model is accessible (list_models is lightweight)
            genai.list_models()
            logger.info(f"Gemini client initialized successfully with model: {self.gemini_model_name}")
        except Exception as e:
            logger.exception(f"Failed to initialize Gemini client with model {self.gemini_model_name}")
            self.console.print(f"[bold red]Error initializing Gemini:[/bold red] {e}")
            self.console.print("Please check your API key, model name, and network connection.")
            raise

    def _initialize_tokenizer(self):
        """Initialize tiktoken tokenizer."""
        try:
            # Use a common tokenizer like cl100k_base or gpt-4 as a proxy
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Tiktoken tokenizer initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize tiktoken tokenizer: {e}. History token counting will be approximate.")
            self._tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        if self._tokenizer and text:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                # Fallback on error
                pass
        # Fallback: roughly 4 chars per token
        return len(text) // 4

    def _get_history_token_count(self) -> int:
        """Calculate the total token count of the conversation history."""
        total_tokens = 0
        if not self._tokenizer:
             # Simple approximation if no tokenizer
             return sum(len(part.text) for content in self.conversation_history for part in content.parts if hasattr(part, 'text')) // 4

        for content in self.conversation_history:
            for part in content.parts:
                 if hasattr(part, 'text') and part.text:
                     total_tokens += self._count_tokens(part.text)
                 # Add approximation for function calls/responses if needed
                 elif hasattr(part, 'function_call'):
                     # Crude approximation for function calls/responses
                     total_tokens += self._count_tokens(json.dumps({"name": part.function_call.name, "args": dict(part.function_call.args or {})})) + 10
                 elif hasattr(part, 'function_response'):
                     total_tokens += self._count_tokens(json.dumps({"name": part.function_response.name, "response": dict(part.function_response.response or {})})) + 10
        return total_tokens

    def _truncate_history(self):
        """Truncate conversation history if it exceeds the token limit."""
        current_tokens = self._get_history_token_count()
        if current_tokens > self.max_history_tokens:
            logger.warning(f"History token count ({current_tokens}) exceeds limit ({self.max_history_tokens}). Truncating.")
            tokens_to_remove = current_tokens - self.max_history_tokens
            removed_tokens = 0
            # Remove from the beginning, skipping the first two (system prompt + initial model ack)
            while removed_tokens < tokens_to_remove and len(self.conversation_history) > 2:
                content_to_remove = self.conversation_history.pop(0) # Remove oldest first
                for part in content_to_remove.parts:
                     if hasattr(part, 'text') and part.text:
                         removed_tokens += self._count_tokens(part.text)
                     # Add approximations for removed function calls/responses
                     elif hasattr(part, 'function_call'):
                        removed_tokens += self._count_tokens(json.dumps({"name": part.function_call.name, "args": dict(part.function_call.args or {})})) + 10
                     elif hasattr(part, 'function_response'):
                        removed_tokens += self._count_tokens(json.dumps({"name": part.function_response.name, "response": dict(part.function_response.response or {})})) + 10

            new_tokens = self._get_history_token_count()
            logger.info(f"History truncated. Removed ~{removed_tokens} tokens. New count: {new_tokens}")


    def _create_system_prompt(self) -> str:
        """Creates the system prompt for Gemini."""
        # Note: Tool descriptions will be provided via the 'tools' parameter in the API call
        # This prompt focuses on the *behavior* and *workflow*.
        return f"""You are an AI assistant designed to interact with a user's development environment via a set of provided tools managed by MCP (Message Control Protocol) servers. Your goal is to understand the user's request, plan the necessary steps (if complex), and utilize the available tools effectively to fulfill the request.

Available Tool Types (Examples):
* `list_directory`: View files and folders.
* `read_file`: Get the content of a file.
* `edit_file`: Create, overwrite, or append to files. **REQUIRES `confirm: true` argument to execute.**
* `git_operation`: Perform Git actions (status, diff, log, etc.).
* `run_command`: Execute shell commands.
* Other domain-specific tools provided by connected MCP servers.

Workflow:
1.  **Analyze & Plan:** Understand the user's request. For simple, single-tool requests (e.g., "list files", "show me foo.py"), proceed directly to execution. For complex tasks requiring multiple steps (e.g., "refactor function X in file Y, then run tests"), briefly outline your plan in text first *before* making the first tool call.
2.  **Execute:** Use the appropriate tool by making a function call. You will receive the result back. **ALWAYS use tools directly when an action is requested.** Do not just say you will do something, *do it* by calling the tool. For `edit_file`, remember `confirm: true`.
3.  **Observe & Iterate:** Analyze the tool's result. If the task requires further steps, make the next necessary function call. Continue this process until the user's goal is achieved. You can make up to {self.max_tool_calls_per_turn} tool calls consecutively if needed for a single user request.
4.  **Complete & Summarize:** Once the *entire* request is fulfilled, provide a final response to the user summarizing what was done. Crucially, **end your final response with the exact marker:** `{TASK_COMPLETE_MARKER}` followed by a concise summary formatted in Markdown.

Important Rules:
* **Direct Action:** Prioritize using tools immediately over discussing plans for simple requests. If the user asks "what's in this directory?", your first action MUST be to call `list_directory`. If they ask "show file.txt", your first action MUST be `read_file`.
* **Tool Usage:** Only use the tools provided to you via function calls. Do not invent tools or arguments.
* **`edit_file` Confirmation:** You MUST include `"confirm": true` in the arguments for `edit_file` to actually modify files. The user may be prompted to confirm this action by the client.
* **Sequential Calls:** You can make multiple tool calls in sequence within a single turn if necessary to complete a task, up to the limit of {self.max_tool_calls_per_turn}.
* **Contextual Awareness:** Use the conversation history provided to understand the context of the user's requests.
* **Task Completion Signal:** ALWAYS signal the end of fulfilling a user's request by appending `{TASK_COMPLETE_MARKER} <Your Markdown Summary>` to your very last message for that request. If you cannot complete the task, explain why and still use the marker.
* **Clarity:** Explain tool results clearly to the user in Markdown format. For file content, use appropriate code blocks.
"""


    async def connect_to_all_servers(self):
        """Connect to all configured MCP servers."""
        if not self.mcp_servers_config:
            self.console.print("[yellow]No MCP servers defined in the configuration.[/yellow]")
            self.console.print("You can import from Claude Desktop config using `mcp-client import-claude-config` or manually edit the config file.")
            return

        self.console.print("Connecting to MCP servers...")
        connected_count = 0
        for name, config in self.mcp_servers_config.items():
            with self.console.status(f"Connecting to {name}...") as status:
                try:
                    await self.connect_to_server(name, config)
                    status.update(f"[green]Connected to {name}[/green]")
                    connected_count += 1
                except Exception as e:
                    logger.error(f"Failed to connect to server {name}: {str(e)}", exc_info=True)
                    status.update(f"[red]Failed to connect to {name}: {e}[/red]")
        self.console.print(f"Finished connecting. ({connected_count}/{len(self.mcp_servers_config)} servers connected)")


    async def connect_to_server(self, name: str, config: McpServerConfig):
        """Connect to a specific MCP server."""
        logger.info(f"Attempting connection to MCP server: {name} (Cmd: {config.command} {' '.join(config.args)})")

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        await session.initialize() # Handshake

        self.sessions[name] = session
        logger.info(f"Successfully connected and initialized session for: {name}")

        # Log available tools/resources (optional, can be chat command)
        # await self.list_server_details(name, session)


    async def list_server_details(self, server_name: str, session: ClientSession):
        """Lists tools and resources for a given server session."""
        details = [f"[bold]Server '{server_name}':[/bold]"]
        try:
            tools_response = await session.list_tools()
            if tools_response.tools:
                 details.append("  [cyan]Tools:[/cyan]")
                 for tool in tools_response.tools:
                    # Clean schema for display if needed (optional)
                    # input_schema_str = json.dumps(self._prepare_tool_schema(tool.inputSchema), indent=2)
                    details.append(f"    - [bold]{tool.name}[/bold]: {tool.description or '(No description)'}")
                    # details.append(f"      Schema: {input_schema_str}") # Can be very verbose
            else:
                details.append("  (No tools found)")
        except Exception as e:
            logger.error(f"Failed to list tools for server {server_name}: {str(e)}", exc_info=True)
            details.append(f"  [red]Error listing tools: {e}[/red]")

        # Add resource listing if needed (similar structure)

        return "\n".join(details)


    def _prepare_tool_schema(self, tool_schema: Any) -> Any:
        """
        Clean up the MCP tool schema recursively to be compatible with Gemini.
        Removes 'default' fields.
        """
        if isinstance(tool_schema, dict):
            return {
                key: self._prepare_tool_schema(value)
                for key, value in tool_schema.items()
                if key != 'default' # Gemini often errors on 'default'
            }
        elif isinstance(tool_schema, list):
            return [self._prepare_tool_schema(item) for item in tool_schema]
        return tool_schema

    async def _get_available_tools(self) -> Tuple[List[genai_types.Tool], Dict[str, str]]:
        """Fetch tools from all connected servers and format for Gemini."""
        all_tool_declarations = []
        server_tools_map: Dict[str, str] = {} # tool_name -> server_name

        if not self.sessions:
             return [], {}

        tasks = []
        for server_name, session in self.sessions.items():
            tasks.append(session.list_tools()) # Collect coroutines

        tool_responses = await asyncio.gather(*tasks, return_exceptions=True)

        server_names = list(self.sessions.keys())
        for i, response in enumerate(tool_responses):
            server_name = server_names[i]
            if isinstance(response, Exception):
                logger.error(f"Failed to list tools for server {server_name}: {response}")
                continue # Skip this server if listing failed

            if hasattr(response, 'tools'):
                for tool in response.tools:
                    if not tool.name:
                        logger.warning(f"Skipping tool with no name from server {server_name}")
                        continue

                    server_tools_map[tool.name] = server_name
                    try:
                        # Clean schema for Gemini compatibility
                        cleaned_schema = self._prepare_tool_schema(tool.inputSchema or {})

                        # Ensure parameters is a valid schema object, not just {}
                        parameters_schema = None
                        if cleaned_schema and isinstance(cleaned_schema, dict) and cleaned_schema.get("properties"):
                             parameters_schema=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties=cleaned_schema.get("properties", {}),
                                required=cleaned_schema.get("required", None)
                             )

                        declaration = genai_types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description or f"Tool: {tool.name}",
                            parameters=parameters_schema
                        )
                        all_tool_declarations.append(declaration)
                    except Exception as e:
                         logger.error(f"Error processing tool '{tool.name}' from server '{server_name}': {e}", exc_info=True)


        if not all_tool_declarations:
             logger.warning("No valid tools found across all connected servers.")
             return [], {}

        # Gemini expects tools wrapped in a Tool object
        gemini_tools = [genai_types.Tool(function_declarations=all_tool_declarations)]
        return gemini_tools, server_tools_map

    async def process_query(self, query: str) -> str:
        """Process a user query using Gemini and MCP tools with multi-step orchestration."""

        if not self.sessions:
            return "[bold red]Error:[/bold red] No MCP servers connected."
        if not self.gemini_client:
             return "[bold red]Error:[/bold red] Gemini client not initialized."

        # Add user query to history (unless it's empty, which implies continuation)
        if query:
            user_content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
            self.conversation_history.append(user_content)
        elif not self.conversation_history: # Handle empty initial query
             query = "List the files in the current directory." # Default action
             user_content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
             self.conversation_history.append(user_content)

        # Ensure history doesn't exceed limits before API call
        self._truncate_history()

        gemini_tools, server_tools_map = await self._get_available_tools()
        if not gemini_tools:
             return "[bold yellow]Warning:[/bold yellow] No tools available from connected MCP servers."

        current_iteration = 0
        final_response_text = ""

        try:
            while current_iteration < MAX_AGENT_ITERATIONS:
                current_iteration += 1
                logger.info(f"--- Agent Iteration {current_iteration} ---")

                # Prepare request for Gemini
                request_contents = self.conversation_history.copy() # Use current state of history
                temperature = self.config.get("gemini", {}).get("temperature", 0.1)
                gen_config = genai_types.GenerationConfig(
                    temperature=temperature,
                    # candidate_count=1, # Default is 1
                    # max_output_tokens=..., # Optional limit
                )

                llm_response: Optional[genai_types.GenerateContentResponse] = None
                with self.console.status(f"[yellow]Assistant thinking (Iteration {current_iteration})...[/yellow]", spinner="dots"):
                    try:
                        llm_response = await self.gemini_client.generate_content(
                            contents=request_contents,
                            generation_config=gen_config,
                            tools=gemini_tools,
                        )
                    except google_exceptions.ResourceExhausted as e:
                        logger.error(f"Gemini API Quota Exceeded: {e}")
                        return f"[bold red]Error:[/bold red] Gemini API quota exceeded. Please check your usage limits. ({e})"
                    except Exception as e:
                        logger.exception("Error calling Gemini API")
                        return f"[bold red]Error:[/bold red] Failed to get response from Gemini: {e}"

                if not llm_response or not llm_response.candidates:
                    logger.error("Received empty response or no candidates from Gemini.")
                    return "[bold red]Error:[/bold red] Received no response from Gemini."

                candidate = llm_response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                     logger.warning("Gemini response candidate has no content/parts.")
                     # If there was a previous text response, return that. Otherwise, error.
                     return final_response_text or "[bold yellow]Warning:[/bold yellow] Gemini returned an empty response."

                # Add the model's response (potentially including function calls) to history
                self.conversation_history.append(candidate.content)
                self._truncate_history() # Truncate *after* adding

                # Process parts: collect text and function calls
                response_text_parts = []
                function_calls_to_execute = []

                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text_parts.append(part.text)
                    elif hasattr(part, 'function_call') and part.function_call.name:
                        # Check against tool call limit for this turn
                        if len(function_calls_to_execute) < self.max_tool_calls_per_turn:
                             function_calls_to_execute.append(part.function_call)
                        else:
                             logger.warning(f"Exceeded max tool calls per turn ({self.max_tool_calls_per_turn}). Skipping further calls in this iteration.")
                             # Potentially add a message to the user/LLM about skipping calls

                current_response_text = "\n".join(response_text_parts).strip()
                if current_response_text:
                     # Keep track of the latest text response
                     final_response_text = current_response_text
                     # Display intermediate thoughts/plans from the model
                     self.console.print("[cyan]Assistant:[/cyan]")
                     self.console.print(Syntax(current_response_text, "markdown", theme="default", word_wrap=True))


                # --- Check for Task Completion Marker ---
                if TASK_COMPLETE_MARKER in final_response_text:
                    logger.info("Task complete marker detected.")
                    # Extract summary after marker
                    summary = final_response_text.split(TASK_COMPLETE_MARKER, 1)[-1].strip()
                    # Return only the summary part
                    return summary or "Task completed." # Return summary or default message

                # --- Execute Function Calls ---
                if not function_calls_to_execute:
                    logger.info("No function calls requested in this iteration. Assuming task completion or needs clarification.")
                    # If no function calls, and no completion marker, the loop ends, return the last text.
                    return final_response_text or "[italic]Assistant provided no further action or summary.[/italic]"

                # --- Execute Tools and Prepare Function Responses ---
                function_responses_for_gemini = []
                executed_tool_results_for_user = []

                for func_call in function_calls_to_execute:
                    tool_name = func_call.name
                    arguments = dict(func_call.args or {}) # Convert Struct to dict

                    self.console.print(f"\n[yellow]🔧 Tool Call Requested:[/yellow] [bold]{tool_name}[/bold]")
                    self.console.print(f"   [dim]Arguments: {json.dumps(arguments)}[/dim]")

                    if tool_name not in server_tools_map:
                        logger.error(f"Tool '{tool_name}' requested but not found in any connected server map.")
                        result_text = f"Error: Tool '{tool_name}' is not available."
                        executed_tool_results_for_user.append(f"[red]Error:[/red] Tool '{tool_name}' not found.")
                    else:
                        server_name = server_tools_map[tool_name]
                        session = self.sessions.get(server_name)
                        if not session:
                             logger.error(f"Server '{server_name}' for tool '{tool_name}' not connected.")
                             result_text = f"Error: Server '{server_name}' for tool '{tool_name}' is not connected."
                             executed_tool_results_for_user.append(f"[red]Error:[/red] Server '{server_name}' disconnected.")
                        else:
                            # --- Confirmation for Risky Tools ---
                            confirmed = True
                            if self.confirm_risky_tools and tool_name in ["edit_file", "delete_path", "run_command"]: # Add other risky tools
                                try:
                                    confirm_msg = f"Allow assistant to execute '{tool_name}' with args: {json.dumps(arguments)}?"
                                    confirmed = questionary.confirm(confirm_msg, default=False, auto_enter=False).ask()
                                    if confirmed is None: # User pressed Ctrl+C
                                        self.console.print("[yellow]Tool execution cancelled by user.[/yellow]")
                                        return "Operation cancelled."
                                except Exception as q_err:
                                    logger.error(f"Questionary confirmation failed: {q_err}")
                                    confirmed = False # Fail safe

                            if not confirmed:
                                result_text = f"User rejected execution of tool '{tool_name}'."
                                executed_tool_results_for_user.append(f"[yellow]Skipped:[/yellow] User rejected execution of '{tool_name}'.")
                                self.console.print(f"[yellow]User rejected execution of {tool_name}.[/yellow]")
                            else:
                                # --- Actual Tool Execution ---
                                with self.console.status(f"[yellow]Executing {tool_name} on {server_name}...[/yellow]", spinner="dots"):
                                     result_text = await self._execute_tool(session, tool_name, arguments)

                                # Format result for user display (append after loop)
                                formatted_result = self._format_tool_result_for_user(result_text, tool_name, arguments)
                                executed_tool_results_for_user.append(formatted_result)

                    # --- Prepare FunctionResponse for Gemini ---
                    # Gemini expects the result in a specific structure
                    # Ensure the result is JSON serializable; often just needs to be a string.
                    serializable_result = {"result": str(result_text)} # Ensure it's a dict with a key

                    function_response_part = genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name,
                            response=serializable_result
                        )
                    )
                    function_responses_for_gemini.append(function_response_part)

                # Display formatted tool results to user
                if executed_tool_results_for_user:
                    self.console.print("\n[green]=> Tool Results:[/green]")
                    for res in executed_tool_results_for_user:
                        self.console.print(res) # Should already be formatted with Panels/Syntax

                # Add function results to history for the *next* Gemini call
                if function_responses_for_gemini:
                    # The role is 'tool' according to Gemini docs for function responses
                    tool_response_content = genai_types.Content(role="tool", parts=function_responses_for_gemini)
                    self.conversation_history.append(tool_response_content)
                    self._truncate_history() # Truncate after adding results

                # Loop continues: Gemini will get the tool results in the history

            # If loop finishes due to max iterations
            logger.warning(f"Reached max agent iterations ({MAX_AGENT_ITERATIONS}).")
            final_response_text += f"\n\n[bold yellow]Warning:[/bold yellow] Reached maximum interaction limit ({MAX_AGENT_ITERATIONS}). Task may be incomplete. Please provide further instructions if needed."
            # Append completion marker here? Or let user decide? Let's add it.
            final_response_text += f"\n\n{TASK_COMPLETE_MARKER} Max iterations reached."
            return final_response_text

        except Exception as e:
            logger.exception("Error during agent processing loop")
            return f"[bold red]An unexpected error occurred:[/bold red] {e}"


    async def _execute_tool(self, session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool and return its result as text."""
        try:
            # Ensure confirm=true for edits if not explicitly denied by user earlier
            if tool_name == "edit_file" and arguments.get("confirm") is not False:
                arguments["confirm"] = True
                logger.info("Ensuring 'confirm: true' for edit_file tool.")


            tool_result = await session.call_tool(tool_name, arguments)
            result_text = ""

            # Extract text content from the tool result
            if hasattr(tool_result, 'content') and tool_result.content:
                result_text = "".join(
                    item.text for item in tool_result.content if hasattr(item, 'text')
                ).strip()

            if tool_result.isError:
                logger.error(f"MCP Tool '{tool_name}' error: {result_text}")
                # Prepend Error for clarity
                return f"Error executing tool '{tool_name}': {result_text}"
            else:
                logger.info(f"MCP Tool '{tool_name}' executed successfully.")
                log_preview = result_text[:200] + ('...' if len(result_text) > 200 else '')
                logger.debug(f"Tool '{tool_name}' result preview: {log_preview}")
                return result_text or "(Tool executed successfully with no text output)"

        except Exception as e:
            logger.exception(f"Error calling MCP tool {tool_name}")
            return f"Error calling tool {tool_name}: {str(e)}"

    def _format_tool_result_for_user(self, result_text: str, tool_name: str, arguments: Dict) -> Any:
        """Format tool results using Rich components for display."""
        from rich.panel import Panel

        # Check for errors first
        is_error = result_text.lower().startswith("error:")
        title = f"Result: [bold]{tool_name}[/bold]"
        border_style = "red" if is_error else "blue"

        # Specific formatting based on tool
        if not is_error and tool_name == "list_directory":
            # Use code block for directory listing
            syntax = Syntax(result_text, "bash", theme="default", line_numbers=False, word_wrap=True)
            return Panel(syntax, title=title, border_style=border_style, expand=False)
        elif not is_error and tool_name == "read_file":
             file_path = arguments.get('file_path', 'file')
             ext = os.path.splitext(file_path)[1].lower().lstrip('.')
             # Guess lexer, fallback to text
             lexer = ext if ext else "text"
             try:
                 syntax = Syntax(result_text, lexer, theme="default", line_numbers=True, word_wrap=True)
                 return Panel(syntax, title=f"Content: [bold]{file_path}[/bold]", border_style=border_style, expand=False)
             except Exception: # Handle invalid lexer
                 return Panel(result_text, title=f"Content: [bold]{file_path}[/bold]", border_style=border_style, expand=False)
        else:
            # Default: just put the text in a panel
            # Use syntax highlighting for bash commands if possible
            if tool_name == "run_command":
                 syntax = Syntax(result_text, "bash", theme="default", line_numbers=False, word_wrap=True)
                 return Panel(syntax, title=title, border_style=border_style, expand=False)
            else:
                 return Panel(result_text, title=title, border_style=border_style, expand=False)


    async def cleanup(self):
        """Clean up resources."""
        self.console.print("\nCleaning up connections...")
        await self.exit_stack.aclose()
        self.console.print("Cleanup complete.")