"""
Configuration management for MCP Client.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Use the logger configured in main.py
logger = logging.getLogger("mcp_client.config") # More specific logger name

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_MAX_HISTORY_TOKENS = 50000

class McpServerConfig:
    """Configuration for an MCP server."""
    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}

class Config:
    """Manages configuration for the MCP Client application."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path:
            self.config_file = config_path
            self.config_dir = config_path.parent
            logger.info(f"Using specified config path: {self.config_file}")
        else:
            self.config_dir = Path.home() / ".config" / "mcp-client"
            self.config_file = self.config_dir / "config.yaml"
            logger.info(f"Using default config path: {self.config_file}")

        self._ensure_config_exists()
        self.config_data = self._load_config()
        if not self.config_data:
             logger.warning(f"Configuration data is empty after loading {self.config_file}.")
             # Attempt to ensure defaults again if loading failed badly
             self._ensure_config_exists(force_defaults_if_empty=True)
             self.config_data = self._load_config() # Try loading again


    def _ensure_config_exists(self, force_defaults_if_empty=False):
        """Create config directory and file if they don't exist or if forced."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured config directory exists: {self.config_dir}")

            if not self.config_file.exists() or force_defaults_if_empty:
                if not self.config_file.exists():
                    logger.info(f"Config file not found at {self.config_file}. Creating default.")
                else:
                     logger.warning(f"Forcing default config creation for {self.config_file}.")

                default_config = {
                    "gemini": {
                        "api_key": None,
                        "default_model": DEFAULT_MODEL,
                        "temperature": 0.1,
                    },
                    "mcp_servers": {},
                    "settings": {
                        "log_level": "INFO",
                        "confirm_risky_tools": True,
                        "max_history_tokens": DEFAULT_MAX_HISTORY_TOKENS,
                        "max_tool_calls_per_turn": 5,
                    }
                }
                # Only save defaults if the file truly didn't exist OR force is true
                # Avoid overwriting potentially corrupt file unless forced
                if not self.config_file.exists() or force_defaults_if_empty:
                    self._save_config(default_config)
                    logger.info(f"Default config file written to {self.config_file}")
                else:
                     logger.warning(f"Config file {self.config_file} exists but load resulted in empty data. NOT overwriting with defaults unless forced.")


        except OSError as e:
             logger.error(f"Error creating config directory or file at {self.config_file}: {e}", exc_info=True)
             # Re-raise or handle as appropriate for your application context
             # raise


    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
             logger.warning(f"Cannot load config, file does not exist: {self.config_file}")
             return {}
        try:
            with open(self.config_file, 'r') as f:
                loaded_data = yaml.safe_load(f)
                if loaded_data is None:
                     logger.warning(f"Config file {self.config_file} is empty or invalid YAML, loaded as None.")
                     return {}
                logger.info(f"Successfully loaded config from {self.config_file}")
                return loaded_data
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {self.config_file}: {e}", exc_info=True)
            return {}
        except OSError as e:
             logger.error(f"Error reading config file {self.config_file}: {e}", exc_info=True)
             return {}


    def _save_config(self, data: Optional[Dict[str, Any]] = None):
        """Save configuration to file."""
        save_data = data if data is not None else self.config_data
        if not save_data: # Avoid writing empty data unless explicitly asked
             logger.warning("Attempted to save empty or None config data. Skipping save.")
             # Optionally raise an error or return False?
             return

        try:
            # Ensure directory exists before writing
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(save_data, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except OSError as e:
            logger.error(f"Error saving config file {self.config_file}: {e}", exc_info=True)
        except yaml.YAMLError as e:
             logger.error(f"Error dumping YAML data for {self.config_file}: {e}", exc_info=True)


    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value."""
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any):
        """Set a top-level config value and save."""
        self.config_data[key] = value
        self._save_config()

    def get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key from config or environment."""
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            # logger.debug("Using Gemini API key from environment variable.")
            return env_key
        config_key = self.config_data.get("gemini", {}).get("api_key")
        # if config_key:
        #      logger.debug("Using Gemini API key from config file.")
        # else:
        #      logger.warning("Gemini API key not found in environment or config file.")
        return config_key


    def set_gemini_api_key(self, key: str):
        """Set Gemini API key in the config file."""
        if not isinstance(self.config_data.get("gemini"), dict):
            self.config_data["gemini"] = {}
        self.config_data["gemini"]["api_key"] = key
        self._save_config()
        logger.info("Gemini API key updated in config file.")


    def get_default_model(self) -> str:
        """Get the default Gemini model."""
        return self.config_data.get("gemini", {}).get("default_model", DEFAULT_MODEL)

    def set_default_model(self, model_id: str):
        """Set the default Gemini model."""
        if not isinstance(self.config_data.get("gemini"), dict): self.config_data["gemini"] = {}
        self.config_data["gemini"]["default_model"] = model_id # Corrected key
        self._save_config()
        logger.info(f"Default Gemini model set to '{model_id}' in config.")


    def get_setting(self, setting: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        return self.config_data.get("settings", {}).get(setting, default)

    def set_setting(self, setting: str, value: Any):
        """Set a specific setting value."""
        if not isinstance(self.config_data.get("settings"), dict): self.config_data["settings"] = {}
        self.config_data["settings"][setting] = value
        self._save_config()
        logger.info(f"Setting '{setting}' updated to '{value}' in config.")


    def get_mcp_servers(self) -> Dict[str, McpServerConfig]:
        """Get MCP server configurations."""
        servers_dict = self.config_data.get("mcp_servers", {})
        if not isinstance(servers_dict, dict):
             logger.error("Invalid 'mcp_servers' format in config, expected a dictionary.")
             return {}

        servers = {}
        for name, data in servers_dict.items():
             if not isinstance(data, dict):
                  logger.warning(f"Skipping invalid MCP server config entry for '{name}': not a dictionary.")
                  continue
             try:
                servers[name] = McpServerConfig(
                    name=name,
                    command=data.get("command", ""),
                    args=data.get("args", []),
                    env=data.get("env", {})
                )
             except Exception as e:
                logger.warning(f"Skipping invalid MCP server config data for '{name}': {e}")
        logger.debug(f"Loaded {len(servers)} MCP server configurations.")
        return servers


    def set_mcp_servers(self, servers: Dict[str, McpServerConfig]):
        """Set MCP server configurations."""
        self.config_data["mcp_servers"] = {
            name: {"command": s.command, "args": s.args, "env": s.env}
            for name, s in servers.items()
        }
        self._save_config()
        logger.info(f"Set {len(servers)} MCP server configurations in config.")


    def add_mcp_server(self, server: McpServerConfig):
        """Add or update an MCP server configuration."""
        if not isinstance(self.config_data.get("mcp_servers"), dict):
             self.config_data["mcp_servers"] = {}
        self.config_data["mcp_servers"][server.name] = {
            "command": server.command,
            "args": server.args,
            "env": server.env
        }
        self._save_config()
        logger.info(f"Added/Updated MCP server '{server.name}' in config.")


    def import_claude_desktop_config(self) -> bool:
        """Attempt to import MCP server configs from Claude Desktop config."""
        # (Implementation remains the same as previously provided)
        config_path_str = None
        possible_paths = [
            os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json"), # macOS
            os.path.join(os.getenv("APPDATA", ""), "Claude", "claude_desktop_config.json"), # Windows
            # Add Linux path if known, e.g., ~/.config/Claude/claude_desktop_config.json
             Path.home() / ".config" / "Claude" / "claude_desktop_config.json", # Common Linux path
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_path_str = path
                break

        if not config_path_str:
            logger.warning("Claude Desktop config not found at common locations.")
            return False

        logger.info(f"Found Claude Desktop config at: {config_path_str}")
        try:
            with open(config_path_str, 'r') as f:
                claude_config = json.load(f)

            imported_servers = {}
            count = 0
            if "mcpServers" in claude_config:
                for name, server_config in claude_config["mcpServers"].items():
                    try:
                         server = McpServerConfig(
                            name=name,
                            command=server_config.get("command", ""),
                            args=server_config.get("args", []),
                            env=server_config.get("env", {})
                        )
                         if not server.command:
                             logger.warning(f"Skipping server '{name}' from Claude config: Missing command.")
                             continue

                         self.add_mcp_server(server)
                         count += 1
                         # No need for separate log here, add_mcp_server logs
                    except Exception as e:
                        logger.warning(f"Skipping invalid MCP server config for '{name}' in Claude config: {e}")

            if count > 0:
                logger.info(f"Import process complete. Imported/Updated {count} MCP server configurations from Claude Desktop config.")
            else:
                 logger.info("Import process complete. No valid MCP servers found or imported from Claude Desktop config.")
            return count > 0 # Returns True if at least one was imported/updated
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding Claude Desktop config JSON from {config_path_str}: {e}")
            return False
        except OSError as e:
            logger.error(f"Error reading Claude Desktop config file {config_path_str}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error importing Claude Desktop config: {e}", exc_info=True)
            return False