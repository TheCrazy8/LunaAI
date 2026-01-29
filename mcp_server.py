"""
MCP Server Integration for LunaAI
Provides Model Context Protocol server support for external integrations.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """MCP request structure."""
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class MCPResponse:
    """MCP response structure."""
    result: Any
    id: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class LunaAIMCPServer:
    """
    MCP Server for LunaAI.
    Provides standardized interface for external tools and integrations.
    """
    
    def __init__(self, model_path: str = "./luna_model", host: str = "127.0.0.1", port: int = 8765):
        """
        Initialize the MCP server.
        
        Args:
            model_path: Path to the trained model
            host: Server host address
            port: Server port
        """
        self.model_path = model_path
        self.host = host
        self.port = port
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.conversation_contexts: Dict[str, List[str]] = {}
        
        logger.info(f"Initialized MCP Server at {host}:{port}")
    
    async def load_model(self):
        """Load the AI model asynchronously."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
            )
            self.model.eval()
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle incoming MCP requests.
        
        Args:
            request: MCP request object
            
        Returns:
            MCP response object
        """
        try:
            if not self.model_loaded:
                return MCPResponse(
                    result=None,
                    id=request.id,
                    error={"code": -1, "message": "Model not loaded"}
                )
            
            method = request.method
            params = request.params
            
            if method == "generate":
                result = await self._handle_generate(params)
            elif method == "chat":
                result = await self._handle_chat(params)
            elif method == "get_model_info":
                result = await self._handle_get_model_info(params)
            elif method == "clear_context":
                result = await self._handle_clear_context(params)
            else:
                return MCPResponse(
                    result=None,
                    id=request.id,
                    error={"code": -2, "message": f"Unknown method: {method}"}
                )
            
            return MCPResponse(result=result, id=request.id)
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return MCPResponse(
                result=None,
                id=request.id,
                error={"code": -3, "message": str(e)}
            )
    
    async def _handle_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle text generation request.
        
        Args:
            params: Request parameters
            
        Returns:
            Generation result
        """
        prompt = params.get("prompt", "")
        max_length = params.get("max_length", 100)
        temperature = params.get("temperature", 0.7)
        
        if not prompt:
            raise ValueError("Prompt is required")
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "response": response.strip(),
            "tokens_generated": outputs.shape[1] - inputs.shape[1]
        }
    
    async def _handle_chat(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle chat request with context management.
        
        Args:
            params: Request parameters
            
        Returns:
            Chat response
        """
        message = params.get("message", "")
        session_id = params.get("session_id", "default")
        max_length = params.get("max_length", 100)
        
        if not message:
            raise ValueError("Message is required")
        
        # Get or create conversation context
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = []
        
        context = self.conversation_contexts[session_id]
        context.append(f"User: {message}")
        
        # Build prompt with context (last 5 exchanges)
        recent_context = context[-10:]
        prompt = "\n".join(recent_context) + "\nAssistant:"
        
        # Generate response
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + max_length,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Clean up response
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        # Store in context
        context.append(f"Assistant: {response}")
        
        return {
            "message": message,
            "response": response,
            "session_id": session_id,
            "context_length": len(context)
        }
    
    async def _handle_get_model_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            params: Request parameters
            
        Returns:
            Model information
        """
        return {
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "model_type": type(self.model).__name__ if self.model else None,
            "tokenizer_type": type(self.tokenizer).__name__ if self.tokenizer else None,
            "active_sessions": len(self.conversation_contexts)
        }
    
    async def _handle_clear_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clear conversation context.
        
        Args:
            params: Request parameters
            
        Returns:
            Clear result
        """
        session_id = params.get("session_id", "default")
        
        if session_id == "all":
            count = len(self.conversation_contexts)
            self.conversation_contexts.clear()
            return {"cleared": count, "message": f"Cleared all {count} sessions"}
        else:
            if session_id in self.conversation_contexts:
                del self.conversation_contexts[session_id]
                return {"cleared": 1, "message": f"Cleared session: {session_id}"}
            else:
                return {"cleared": 0, "message": f"Session not found: {session_id}"}
    
    async def start_server(self):
        """Start the MCP server."""
        try:
            # Load model first
            await self.load_model()
            
            logger.info(f"Starting MCP server on {self.host}:{self.port}")
            logger.info("MCP Server is running. Supported methods:")
            logger.info("  - generate: Generate text from prompt")
            logger.info("  - chat: Chat with context management")
            logger.info("  - get_model_info: Get model information")
            logger.info("  - clear_context: Clear conversation context")
            
            # Note: In a real implementation, you would start a WebSocket or HTTP server here
            # For this demo, we'll simulate with a simple loop
            logger.info("MCP Server ready (simulated mode)")
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
    
    def process_json_request(self, json_str: str) -> str:
        """
        Process a JSON-RPC style request synchronously.
        
        Args:
            json_str: JSON string request
            
        Returns:
            JSON string response
        """
        try:
            data = json.loads(json_str)
            request = MCPRequest(
                method=data.get("method"),
                params=data.get("params", {}),
                id=data.get("id")
            )
            
            # Run async handler in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.handle_request(request))
            loop.close()
            
            result = {
                "jsonrpc": "2.0",
                "id": response.id
            }
            
            if response.error:
                result["error"] = response.error
            else:
                result["result"] = response.result
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -4, "message": str(e)}
            }
            return json.dumps(error_response, indent=2)


async def main():
    """Main entry point for MCP server."""
    logger.info("="*60)
    logger.info("LunaAI MCP Server")
    logger.info("="*60)
    
    server = LunaAIMCPServer(
        model_path="./luna_model",
        host="127.0.0.1",
        port=8765
    )
    
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
