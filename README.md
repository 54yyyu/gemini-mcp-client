# MCP Client

An enhanced MCP (Message Control Protocol) client using Google Gemini for orchestration.

## Features

- Seamless integration with multiple MCP servers
- Gemini AI orchestration for intelligent tool usage
- Rich console interface with syntax highlighting
- Configuration management with environment variables and config files
- Token-based conversation history management

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-client.git
cd mcp-client

# Install the package
pip install -e .
```

## Configuration

1. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API keys.

3. Optionally import MCP server configurations from Claude Desktop:
   ```bash
   mcp-client import-claude-config
   ```

## Usage

Start an interactive chat session:
```bash
mcp-client chat
```

List configured MCP servers:
```bash
mcp-client list-servers
```

Set your Gemini API key:
```bash
mcp-client setup-gemini YOUR_API_KEY
```

## Requirements

- Python 3.9+
- See `pyproject.toml` for detailed dependencies

## License

MIT
