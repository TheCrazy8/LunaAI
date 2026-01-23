"""
MCP Server Manager for Luna AI
Allows Luna to connect to and use Model Context Protocol servers
"""

import logging
import json
from typing import Dict, List, Any, Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP not available. Install with: pip install mcp")
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


class MCPServerManager:
    """
    Manager for MCP (Model Context Protocol) server connections.
    Allows Luna to access external tools and services through MCP.
    """
    
    def __init__(self):
        self.servers: Dict[str, Any] = {}
        self.active_sessions: Dict[str, Any] = {}
        self.available_tools: Dict[str, List[Dict]] = {}
        logger.info("MCP Server Manager initialized")
    
    def is_available(self) -> bool:
        """Check if MCP is available"""
        return MCP_AVAILABLE
    
    async def connect_server(self, server_name: str, command: str, args: List[str] = None) -> bool:
        """
        Connect to an MCP server.
        
        Args:
            server_name (str): Name to identify this server
            command (str): Command to start the MCP server
            args (list): Arguments for the server command
            
        Returns:
            bool: True if connection successful
        """
        if not MCP_AVAILABLE:
            logger.error("MCP is not available")
            return False
        
        try:
            logger.info(f"Connecting to MCP server: {server_name}")
            
            if args is None:
                args = []
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args
            )
            
            # Store server info
            self.servers[server_name] = {
                'params': server_params,
                'command': command,
                'args': args
            }
            
            logger.info(f"MCP server {server_name} configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MCP server {server_name}: {e}")
            return False
    
    async def list_server_tools(self, server_name: str) -> List[Dict]:
        """
        List available tools from an MCP server.
        
        Args:
            server_name (str): Name of the server
            
        Returns:
            list: List of available tools
        """
        if not MCP_AVAILABLE:
            return []
        
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not found")
            return []
        
        try:
            server_params = self.servers[server_name]['params']
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # List available tools
                    tools_response = await session.list_tools()
                    tools = tools_response.tools if hasattr(tools_response, 'tools') else []
                    
                    # Store tools for this server
                    self.available_tools[server_name] = [
                        {
                            'name': tool.name,
                            'description': tool.description if hasattr(tool, 'description') else '',
                            'input_schema': tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        }
                        for tool in tools
                    ]
                    
                    logger.info(f"Found {len(tools)} tools on server {server_name}")
                    return self.available_tools[server_name]
                    
        except Exception as e:
            logger.error(f"Error listing tools from {server_name}: {e}")
            return []
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on an MCP server.
        
        Args:
            server_name (str): Name of the server
            tool_name (str): Name of the tool to call
            arguments (dict): Arguments for the tool
            
        Returns:
            Any: Tool response
        """
        if not MCP_AVAILABLE:
            return {"error": "MCP not available"}
        
        if server_name not in self.servers:
            return {"error": f"Server {server_name} not found"}
        
        try:
            server_params = self.servers[server_name]['params']
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Call the tool
                    logger.info(f"Calling tool {tool_name} on server {server_name}")
                    result = await session.call_tool(tool_name, arguments=arguments)
                    
                    return result
                    
        except Exception as e:
            error_msg = f"Error calling tool {tool_name} on {server_name}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def get_all_available_tools(self) -> Dict[str, List[Dict]]:
        """Get all available tools from all connected servers"""
        return self.available_tools.copy()
    
    def get_connected_servers(self) -> List[str]:
        """Get list of connected server names"""
        return list(self.servers.keys())
    
    async def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name in self.servers:
            del self.servers[server_name]
            if server_name in self.available_tools:
                del self.available_tools[server_name]
            logger.info(f"Disconnected from MCP server: {server_name}")
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for server_name in list(self.servers.keys()):
            await self.disconnect_server(server_name)
        logger.info("Disconnected from all MCP servers")


class SimpleMCPManager:
    """
    Simple fallback MCP manager when MCP is not available.
    Provides basic server configuration storage.
    """
    
    def __init__(self):
        self.servers: Dict[str, Dict] = {}
        logger.info("Simple MCP Manager initialized (MCP not available)")
    
    def is_available(self) -> bool:
        return False
    
    async def connect_server(self, server_name: str, command: str, args: List[str] = None) -> bool:
        """Store server configuration"""
        self.servers[server_name] = {
            'command': command,
            'args': args or [],
            'status': 'configured (MCP not available)'
        }
        logger.info(f"Server {server_name} configured (MCP not installed)")
        return False
    
    async def list_server_tools(self, server_name: str) -> List[Dict]:
        return []
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return {"error": "MCP not available. Install with: pip install mcp"}
    
    def get_all_available_tools(self) -> Dict[str, List[Dict]]:
        return {}
    
    def get_connected_servers(self) -> List[str]:
        return list(self.servers.keys())
    
    async def disconnect_server(self, server_name: str):
        if server_name in self.servers:
            del self.servers[server_name]
    
    async def disconnect_all(self):
        self.servers.clear()


# Helper function to run async operations in sync context
def run_async(coro):
    """Run an async coroutine in a sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
