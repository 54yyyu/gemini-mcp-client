"""
MCP Client CLI Entry Point using Click and Rich.
"""

import os
import sys
import json
import asyncio
import logging
import signal
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from dotenv import load_dotenv

from . import __version__
from .config import Config
from .client import McpClient

# --- Configuration ---
# Load .env file first if it exists
load_dotenv()

# Setup Rich Console
console = Console()

# Setup Logging
# Determine log level from config or default to INFO
config = Config() # Load config early for logging level
log_level_str = config.get_setting("log_level", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# Configure root logger
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)] # Use RichHandler
)
logger = logging.getLogger(__name__) # Logger for this module

# --- Click CLI Definition ---
@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(__version__, '-v', '--version')
def cli():
    """MCP Client with Google Gemini Orchestration."""
    pass # Base command group does nothing on its own

@cli.command()
@click.option('--model', '-m', default=None, help=f'Gemini model ID to use (overrides config default).')
def chat(model):
    """Start an interactive chat session."""
    console.print(Panel(f"[bold cyan]MCP Client v{__version__}[/bold cyan]\nUsing Gemini for orchestration.", title="Welcome!", border_style="blue"))

    if model:
        config.set_default_model(model)
        console.print(f"Using specified model: [bold]{model}[/bold]")

    try:
        client = McpClient(config, console)

        async def run_chat():
            await client.connect_to_all_servers()
            if not client.sessions:
                 console.print("[yellow]No MCP servers connected. Cannot start chat.[/yellow]")
                 return

            console.print("\nType your query below. Use 'quit' or Ctrl+C to exit, 'help' for commands.")
            while True:
                try:
                    query = await asyncio.to_thread(console.input, "[bold blue]Query> [/bold blue]")
                    query = query.strip()

                    if query.lower() == 'quit':
                        break
                    elif query.lower() == 'help':
                        show_chat_help()
                        continue
                    elif query.lower() == 'servers':
                         if client.sessions:
                             console.print("[bold]Connected Servers:[/bold]")
                             for name in sorted(client.sessions.keys()): console.print(f" - {name}")
                         else:
                             console.print("No servers connected.")
                         continue
                    elif query.lower() == 'tools':
                         await list_all_tools(client)
                         continue
                    elif query.lower() == 'history':
                         show_history(client)
                         continue
                    elif query.lower() == 'clear':
                         client.conversation_history = []
                         console.print("[yellow]Conversation history cleared.[/yellow]")
                         continue


                    if not query: continue # Ignore empty input

                    response = await client.process_query(query)
                    # Response might already be formatted by Rich, print directly if needed,
                    # or use console.print for safety
                    if isinstance(response, str):
                         # Basic check if it seems like markdown before wrapping
                         if "```" in response or "**" in response or "##" in response:
                            from rich.markdown import Markdown
                            console.print(Markdown(response, code_theme="default"))
                         else:
                            console.print(response)
                    else:
                         # Handle Panel/Syntax objects directly
                         console.print(response)


                except (KeyboardInterrupt, EOFError):
                    break
                except Exception as e:
                    logger.exception("Error during chat loop iteration")
                    console.print(f"\n[bold red]Error:[/bold red] {e}")
            await client.cleanup()

        asyncio.run(run_chat())

    except ValueError as e: # Handle config errors like missing API key during init
         console.print(f"[bold red]Configuration Error:[/bold red] {e}")
         sys.exit(1)
    except Exception as e:
        logger.exception("Failed to initialize or run chat session.")
        console.print(f"[bold red]Fatal Error:[/bold red] {e}")
        sys.exit(1)

def show_chat_help():
    help_text = """
[bold]Interactive Chat Commands:[/bold]
  [cyan]quit[/cyan]         Exit the chat session.
  [cyan]help[/cyan]         Show this help message.
  [cyan]servers[/cyan]      List currently connected MCP servers.
  [cyan]tools[/cyan]        List available tools from all connected servers.
  [cyan]history[/cyan]      Show the current conversation history (for debugging).
  [cyan]clear[/cyan]        Clear the current conversation history.

Enter any other text as your query to the assistant.
Use Ctrl+C to exit at any time.
    """
    console.print(Panel(help_text, title="Chat Help", border_style="green", expand=False))

async def list_all_tools(client: McpClient):
     if not client.sessions:
         console.print("No servers connected.")
         return
     console.print("[bold]Available MCP Tools:[/bold]")
     details_list = []
     tasks = []
     for server_name, session in client.sessions.items():
         tasks.append(client.list_server_details(server_name, session))

     results = await asyncio.gather(*tasks, return_exceptions=True)
     for result in results:
         if isinstance(result, Exception):
             console.print(f"[red]Error listing server details: {result}[/red]")
         else:
             console.print(result) # Print details returned by list_server_details


def show_history(client: McpClient):
    console.print(Panel("[bold]Conversation History:[/bold]", border_style="magenta"))
    if not client.conversation_history:
        console.print("[italic]History is empty.[/italic]")
        return

    for i, content in enumerate(client.conversation_history):
        role = content.role.upper()
        color = "blue" if role == "USER" else "cyan" if role == "MODEL" else "yellow"
        console.print(f"[bold {color}]--- {role} Turn {i+1} ---[/bold {color}]")
        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                 from rich.markdown import Markdown
                 console.print(Markdown(part.text, code_theme="default"))
            elif hasattr(part, 'function_call'):
                 fc = part.function_call
                 console.print(f"[yellow]Function Call:[/yellow] [bold]{fc.name}[/bold]")
                 console.print(f"  [dim]Args: {json.dumps(dict(fc.args or {}), indent=2)}[/dim]")
            elif hasattr(part, 'function_response'):
                 fr = part.function_response
                 console.print(f"[yellow]Function Result:[/yellow] [bold]{fr.name}[/bold]")
                 console.print(f"  [dim]Response: {json.dumps(dict(fr.response or {}), indent=2)}[/dim]")
        console.print("-" * 20)


@cli.command()
@click.argument('key', required=True)
def setup_gemini(key):
    """Set and save your Google Gemini API key."""
    try:
        config.set_gemini_api_key(key)
        console.print("[green]✓[/green] Google Gemini API key saved successfully.")
        # Try to initialize client to verify key
        try:
             client_test = McpClient(config, Console(quiet=True)) # Use quiet console for test
             console.print("[green]✓[/green] API key seems valid (client initialized).")
        except Exception as e:
             console.print(f"[yellow]Warning:[/yellow] API key saved, but could not verify immediately: {e}")
    except Exception as e:
        logger.exception("Error saving Gemini API key")
        console.print(f"[bold red]Error:[/bold red] Could not save API key: {e}")
        sys.exit(1)

@cli.command()
def import_claude_config():
    """Import MCP server configurations from Claude Desktop."""
    console.print("Attempting to import MCP server configs from Claude Desktop...")
    success = config.import_claude_desktop_config()
    if success:
        console.print("[green]✓[/green] Successfully imported configurations. They are now saved in the mcp-client config.")
    else:
        console.print("[yellow]![/yellow] Could not import configurations. See logs or ensure Claude Desktop config exists.")

@cli.command()
def list_servers():
    """List MCP servers defined in the configuration."""
    servers = config.get_mcp_servers()
    if not servers:
        console.print("No MCP servers configured.")
        console.print("Use `mcp-client import-claude-config` or edit the config file.")
        return
    console.print("[bold]Configured MCP Servers:[/bold]")
    for name, s_config in servers.items():
        console.print(f"- [cyan]{name}[/cyan]")
        console.print(f"  Command: `{s_config.command}`")
        console.print(f"  Args: `{s_config.args}`")
        console.print(f"  Env: `{s_config.env}`")

# --- Main Execution & Signal Handling ---
def handle_sigint(signum, frame):
    console.print("\n[yellow]Interrupt received, shutting down gracefully...[/yellow]")
    # Note: This won't immediately stop asyncio tasks, relies on loop breaks/cleanup
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    cli()