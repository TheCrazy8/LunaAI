"""
Luna AI Model - Intelligent AI assistant that uses HuggingFace datasets
"""

import logging
from typing import List, Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
except ImportError:
    logger.warning("transformers or torch not available. Install with: pip install transformers torch")
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    torch = None

try:
    from mcp_manager import MCPServerManager, SimpleMCPManager
except ImportError:
    logger.warning("MCP manager not available")
    MCPServerManager = None
    SimpleMCPManager = None


class LunaAI:
    """
    Luna - An AI model that intelligently queries and uses HuggingFace datasets
    """
    
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct"):
        """
        Initialize Luna AI model.
        
        Args:
            model_name (str): HuggingFace model to use for Luna
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        
        # Initialize MCP manager
        if MCPServerManager:
            self.mcp_manager = MCPServerManager()
        elif SimpleMCPManager:
            self.mcp_manager = SimpleMCPManager()
        else:
            self.mcp_manager = None
        
        logger.info(f"Luna AI initialized with model: {model_name}")
        if self.mcp_manager:
            logger.info(f"MCP support: {'Available' if self.mcp_manager.is_available() else 'Not available'}")
    
    def load_model(self):
        """Load the Luna AI model"""
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        try:
            logger.info(f"Loading Luna AI model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Use CPU or GPU depending on availability
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
            self.is_loaded = True
            logger.info("Luna AI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Luna AI model: {e}")
            raise
    
    def query_dataset(self, query: str, dataset_samples: List[Dict[str, Any]], 
                     dataset_name: str, max_context_items: int = 5) -> str:
        """
        Use Luna AI to intelligently query dataset information.
        
        Args:
            query (str): User's query
            dataset_samples (list): Sample items from the dataset
            dataset_name (str): Name of the dataset being queried
            max_context_items (int): Maximum number of dataset items to include in context
            
        Returns:
            str: Luna's intelligent response
        """
        if not self.is_loaded:
            return "Luna AI model is not loaded. Please load the model first."
        
        try:
            # Prepare context from dataset samples
            context_items = dataset_samples[:max_context_items]
            context_str = self._format_dataset_context(context_items, dataset_name)
            
            # Create prompt for Luna
            prompt = self._create_prompt(query, context_str, dataset_name)
            
            # Generate response using Luna
            logger.info(f"Luna processing query: {query}")
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Extract just the assistant's response
            answer = self._extract_answer(generated_text, prompt)
            
            return answer
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"Luna encountered an error: {error_msg}"
    
    def _format_dataset_context(self, items: List[Dict[str, Any]], dataset_name: str) -> str:
        """Format dataset items into context string"""
        context_parts = [f"Dataset: {dataset_name}\n"]
        
        for i, item in enumerate(items, 1):
            context_parts.append(f"\nSample {i}:")
            
            # Format the item data
            if isinstance(item, dict):
                for key, value in item.items():
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "..."
                    context_parts.append(f"  {key}: {value_str}")
            else:
                item_str = str(item)
                if len(item_str) > 200:
                    item_str = item_str[:200] + "..."
                context_parts.append(f"  {item_str}")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, dataset_name: str) -> str:
        """Create a prompt for Luna AI"""
        prompt = f"""You are Luna, an AI assistant specialized in analyzing and explaining HuggingFace datasets. 
You have access to samples from the {dataset_name} dataset.

Dataset samples:
{context}

User question: {query}

As Luna, provide a helpful, accurate, and insightful answer based on the dataset samples provided. If the samples don't contain enough information to fully answer the question, acknowledge this and provide what insights you can.

Luna's response:"""
        
        return prompt
    
    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """Extract the answer from generated text"""
        # Remove the prompt from the generated text
        if prompt in generated_text:
            answer = generated_text[len(prompt):].strip()
        else:
            # Fallback: look for "Luna's response:" marker
            marker = "Luna's response:"
            if marker in generated_text:
                answer = generated_text.split(marker)[-1].strip()
            else:
                answer = generated_text.strip()
        
        # Clean up the answer
        answer = answer.strip()
        
        # Remove any trailing incomplete sentences
        if answer and not answer[-1] in '.!?':
            # Find the last complete sentence
            last_punct = max(
                answer.rfind('.'),
                answer.rfind('!'),
                answer.rfind('?')
            )
            if last_punct > 0:
                answer = answer[:last_punct + 1]
        
        return answer if answer else "I don't have enough information to answer that question."
    
    def generate_dataset_summary(self, dataset_samples: List[Dict[str, Any]], 
                                 dataset_name: str) -> str:
        """
        Generate an intelligent summary of a dataset using Luna AI.
        
        Args:
            dataset_samples (list): Sample items from the dataset
            dataset_name (str): Name of the dataset
            
        Returns:
            str: Luna's summary of the dataset
        """
        if not self.is_loaded:
            return "Luna AI model is not loaded. Please load the model first."
        
        query = f"Analyze and summarize the structure and content of this dataset. What kind of data does it contain and what might it be used for?"
        return self.query_dataset(query, dataset_samples, dataset_name)
    
    def unload_model(self):
        """Unload the model to free memory"""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            if self.pipeline is not None:
                del self.pipeline
            
            # Clear CUDA cache if available
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info("Luna AI model unloaded")
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")


class SimpleLunaAI:
    """
    Simple fallback version of Luna that works without loading large models.
    Uses pattern matching and basic analysis instead of deep learning.
    """
    
    def __init__(self):
        self.is_loaded = True
        
        # Initialize MCP manager
        if MCPServerManager:
            self.mcp_manager = MCPServerManager()
        elif SimpleMCPManager:
            self.mcp_manager = SimpleMCPManager()
        else:
            self.mcp_manager = None
        
        logger.info("Simple Luna AI initialized (lightweight mode)")
        if self.mcp_manager:
            logger.info(f"MCP support: {'Available' if self.mcp_manager.is_available() else 'Not available'}")
    
    def load_model(self):
        """No model to load for simple version"""
        self.is_loaded = True
        logger.info("Simple Luna AI ready")
    
    def query_dataset(self, query: str, dataset_samples: List[Dict[str, Any]], 
                     dataset_name: str, max_context_items: int = 5) -> str:
        """Simple pattern-based query response"""
        query_lower = query.lower()
        
        # Analyze dataset samples
        num_samples = len(dataset_samples)
        if num_samples == 0:
            return f"Luna: I don't have any samples from the {dataset_name} dataset to analyze."
        
        # Get structure info
        structure_info = []
        if dataset_samples and isinstance(dataset_samples[0], dict):
            keys = list(dataset_samples[0].keys())
            structure_info = keys[:10]  # First 10 fields
        
        response_parts = [f"Luna: Analyzing {dataset_name} dataset..."]
        
        # Answer based on query type
        if any(word in query_lower for word in ['what', 'describe', 'explain', 'about']):
            response_parts.append(f"\nI found {num_samples} samples from this dataset.")
            if structure_info:
                response_parts.append(f"The data contains fields: {', '.join(structure_info)}")
            response_parts.append(f"\nHere are some examples from the dataset:")
            
            # Show a few examples
            for i, sample in enumerate(dataset_samples[:3], 1):
                response_parts.append(f"\nExample {i}:")
                if isinstance(sample, dict):
                    for key, value in list(sample.items())[:5]:
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        response_parts.append(f"  {key}: {value_str}")
        
        elif any(word in query_lower for word in ['count', 'how many', 'number']):
            response_parts.append(f"\nI analyzed {num_samples} samples from {dataset_name}.")
            if structure_info:
                response_parts.append(f"Each sample has {len(structure_info)} main fields.")
        
        elif any(word in query_lower for word in ['structure', 'format', 'schema', 'fields']):
            if structure_info:
                response_parts.append(f"\nThe {dataset_name} dataset has the following structure:")
                response_parts.append(f"Fields: {', '.join(structure_info)}")
            else:
                response_parts.append(f"\nI can see {num_samples} samples, but couldn't determine a clear structure.")
        
        else:
            # General search through samples
            matching_samples = []
            for sample in dataset_samples:
                sample_str = str(sample).lower()
                if any(word in sample_str for word in query_lower.split()):
                    matching_samples.append(sample)
            
            if matching_samples:
                response_parts.append(f"\nI found {len(matching_samples)} relevant samples:")
                for i, sample in enumerate(matching_samples[:3], 1):
                    response_parts.append(f"\nMatch {i}:")
                    if isinstance(sample, dict):
                        for key, value in list(sample.items())[:5]:
                            value_str = str(value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                            response_parts.append(f"  {key}: {value_str}")
            else:
                response_parts.append(f"\nI couldn't find specific matches for '{query}' in the samples.")
                response_parts.append(f"However, I have {num_samples} samples available from {dataset_name}.")
        
        return "\n".join(response_parts)
    
    def generate_dataset_summary(self, dataset_samples: List[Dict[str, Any]], 
                                 dataset_name: str) -> str:
        """Generate a simple summary"""
        if not dataset_samples:
            return f"Luna: No samples available from {dataset_name}."
        
        summary = [f"Luna: Dataset Summary for {dataset_name}"]
        summary.append(f"\nTotal samples analyzed: {len(dataset_samples)}")
        
        if isinstance(dataset_samples[0], dict):
            keys = list(dataset_samples[0].keys())
            summary.append(f"Data fields: {', '.join(keys[:10])}")
            if len(keys) > 10:
                summary.append(f"... and {len(keys) - 10} more fields")
        
        return "\n".join(summary)
    
    def unload_model(self):
        """Nothing to unload"""
        pass
