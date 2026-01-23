"""
Luna AI Training Module
Handles training Luna on HuggingFace datasets
"""

import logging
import os
from typing import List, Dict, Any, Optional
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset, concatenate_datasets
    import torch
    TRAINING_AVAILABLE = True
except ImportError:
    logger.warning("Training dependencies not available. Install: transformers, torch, datasets")
    TRAINING_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    Trainer = None
    DataCollatorForLanguageModeling = None
    Dataset = None
    concatenate_datasets = None
    torch = None

try:
    from luna_cache import LunaCacheManager
except ImportError:
    logger.warning("Luna cache manager not available")
    LunaCacheManager = None


class LunaTrainer:
    """
    Trainer for Luna AI model on HuggingFace datasets
    """
    
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct",
                 output_dir: str = "./luna_trained",
                 use_cache: bool = True):
        """
        Initialize Luna trainer.
        
        Args:
            model_name (str): Base model to fine-tune
            output_dir (str): Directory to save trained model
            use_cache (bool): Whether to use caching
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_data = None
        self.use_cache = use_cache
        
        # Initialize cache manager
        if use_cache and LunaCacheManager:
            self.cache_manager = LunaCacheManager()
            logger.info("Cache manager enabled")
        else:
            self.cache_manager = None
            logger.info("Cache manager disabled")
        
        logger.info(f"Luna Trainer initialized with base model: {model_name}")
    
    def prepare_datasets(self, datasets_dict: Dict[str, Any], 
                        max_samples_per_dataset: int = 1000,
                        force_refresh: bool = False) -> Dataset:
        """
        Prepare and combine datasets for training.
        
        Args:
            datasets_dict (dict): Dictionary of loaded datasets
            max_samples_per_dataset (int): Maximum samples to use from each dataset
            force_refresh (bool): Force refresh cache
            
        Returns:
            Dataset: Combined training dataset
        """
        if not TRAINING_AVAILABLE:
            raise ImportError("Training dependencies not available")
        
        # Check cache first
        cache_key = f"training_data_{len(datasets_dict)}_{max_samples_per_dataset}"
        if self.cache_manager and not force_refresh:
            cached_data = self.cache_manager.get_cached_training_data(cache_key)
            if cached_data is not None:
                logger.info("Using cached training data")
                self.training_data = cached_data
                return cached_data
        
        logger.info(f"Preparing {len(datasets_dict)} datasets for training...")
        
        all_texts = []
        
        for dataset_name, dataset in datasets_dict.items():
            logger.info(f"Processing {dataset_name}...")
            
            # Check if we have cached samples for this dataset
            if self.cache_manager:
                cached_samples = self.cache_manager.get_cached_dataset_samples(dataset_name)
                if cached_samples:
                    logger.info(f"Using {len(cached_samples)} cached samples from {dataset_name}")
                    for item in cached_samples[:max_samples_per_dataset]:
                        text = self._item_to_text(item, dataset_name)
                        if text:
                            all_texts.append({"text": text, "dataset": dataset_name})
                    continue
            
            # Process fresh data
            sample_count = 0
            samples_to_cache = []
            for item in dataset:
                if sample_count >= max_samples_per_dataset:
                    break
                
                # Convert item to text
                text = self._item_to_text(item, dataset_name)
                if text:
                    all_texts.append({"text": text, "dataset": dataset_name})
                    samples_to_cache.append(item)
                    sample_count += 1
            
            # Cache the samples for future use
            if self.cache_manager and samples_to_cache:
                self.cache_manager.cache_dataset_samples(
                    dataset_name,
                    samples_to_cache,
                    {'max_samples': max_samples_per_dataset}
                )
            
            logger.info(f"Collected {sample_count} samples from {dataset_name}")
        
        logger.info(f"Total training samples: {len(all_texts)}")
        
        # Create dataset
        training_dataset = Dataset.from_list(all_texts)
        self.training_data = training_dataset
        
        # Cache the training data
        if self.cache_manager:
            metadata = {
                'num_datasets': len(datasets_dict),
                'max_samples_per_dataset': max_samples_per_dataset,
                'total_samples': len(all_texts),
                'datasets': list(datasets_dict.keys())
            }
            self.cache_manager.cache_training_data(cache_key, training_dataset, metadata)
        
        return training_dataset
    
    def _item_to_text(self, item: Any, dataset_name: str) -> Optional[str]:
        """Convert dataset item to training text"""
        try:
            if isinstance(item, dict):
                # Create a structured text representation
                parts = [f"Dataset: {dataset_name}"]
                
                for key, value in item.items():
                    if value is not None:
                        value_str = str(value)
                        # Limit length
                        if len(value_str) > 500:
                            value_str = value_str[:500] + "..."
                        parts.append(f"{key}: {value_str}")
                
                return "\n".join(parts)
            else:
                # Simple string conversion
                text = str(item)
                if len(text) > 1000:
                    text = text[:1000] + "..."
                return f"Dataset: {dataset_name}\nContent: {text}"
                
        except Exception as e:
            logger.warning(f"Error converting item to text: {e}")
            return None
    
    def train(self, training_dataset: Dataset = None,
             num_epochs: int = 3,
             batch_size: int = 4,
             learning_rate: float = 2e-5,
             save_steps: int = 500,
             max_steps: int = -1) -> bool:
        """
        Train Luna on the prepared datasets.
        
        Args:
            training_dataset (Dataset): Dataset to train on
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate
            save_steps (int): Save checkpoint every N steps
            max_steps (int): Maximum training steps (-1 for full training)
            
        Returns:
            bool: True if training successful
        """
        if not TRAINING_AVAILABLE:
            logger.error("Training dependencies not available")
            return False
        
        if training_dataset is None:
            if self.training_data is None:
                logger.error("No training data available. Call prepare_datasets first.")
                return False
            training_dataset = self.training_data
        
        try:
            logger.info("Loading base model and tokenizer...")
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Tokenize dataset
            logger.info("Tokenizing dataset...")
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )
            
            tokenized_dataset = training_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=training_dataset.column_names
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                save_steps=save_steps,
                save_total_limit=2,
                logging_steps=100,
                max_steps=max_steps,
                report_to="none",  # Disable wandb/tensorboard
                remove_unused_columns=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM, not masked LM
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator
            )
            
            # Train
            logger.info("Starting training...")
            self.trainer.train()
            
            # Save final model
            logger.info(f"Saving trained model to {self.output_dir}")
            self.trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Save training info
            info = {
                "base_model": self.model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "total_samples": len(training_dataset),
                "datasets_used": list(set(training_dataset["dataset"]))
            }
            
            with open(os.path.join(self.output_dir, "training_info.json"), "w") as f:
                json.dump(info, f, indent=2)
            
            # Cache the trained model
            if self.cache_manager:
                model_name = f"luna_trained_{int(time.time())}"
                self.cache_manager.cache_trained_model(model_name, self.output_dir, info)
                logger.info(f"Trained model cached as: {model_name}")
            
            logger.info("Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_trained_model(self, model_path: str = None):
        """Load a trained Luna model"""
        if not TRAINING_AVAILABLE:
            raise ImportError("Training dependencies not available")
        
        if model_path is None:
            model_path = self.output_dir
        
        try:
            logger.info(f"Loading trained model from {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info("Trained model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading trained model: {e}")
            return False


class SimpleLunaTrainer:
    """
    Simple fallback trainer that creates a knowledge base instead of training
    """
    
    def __init__(self, output_dir: str = "./luna_knowledge", use_cache: bool = True):
        self.output_dir = output_dir
        self.knowledge_base = {}
        self.use_cache = use_cache
        
        # Initialize cache manager
        if use_cache and LunaCacheManager:
            self.cache_manager = LunaCacheManager()
            logger.info("Cache manager enabled (simple mode)")
        else:
            self.cache_manager = None
            logger.info("Cache manager disabled (simple mode)")
        
        logger.info("Simple Luna Trainer initialized (knowledge base mode)")
    
    def prepare_datasets(self, datasets_dict: Dict[str, Any],
                        max_samples_per_dataset: int = 1000,
                        force_refresh: bool = False):
        """Prepare knowledge base from datasets"""
        
        # Check cache first
        cache_key = f"simple_kb_{len(datasets_dict)}_{max_samples_per_dataset}"
        if self.cache_manager and not force_refresh:
            cached_kb = self.cache_manager.get_cached_training_data(cache_key)
            if cached_kb is not None:
                logger.info("Using cached knowledge base")
                self.knowledge_base = cached_kb
                return cached_kb
        
        logger.info(f"Building knowledge base from {len(datasets_dict)} datasets...")
        
        for dataset_name, dataset in datasets_dict.items():
            logger.info(f"Processing {dataset_name}...")
            
            # Check for cached samples
            if self.cache_manager:
                cached_samples = self.cache_manager.get_cached_dataset_samples(dataset_name)
                if cached_samples:
                    self.knowledge_base[dataset_name] = {
                        'samples': cached_samples[:max_samples_per_dataset],
                        'count': len(cached_samples[:max_samples_per_dataset])
                    }
                    logger.info(f"Used {len(cached_samples[:max_samples_per_dataset])} cached samples")
                    continue
            
            # Process fresh samples
            samples = []
            for i, item in enumerate(dataset):
                if i >= max_samples_per_dataset:
                    break
                samples.append(item)
            
            self.knowledge_base[dataset_name] = {
                'samples': samples,
                'count': len(samples)
            }
            
            # Cache the samples
            if self.cache_manager and samples:
                self.cache_manager.cache_dataset_samples(
                    dataset_name,
                    samples,
                    {'max_samples': max_samples_per_dataset}
                )
            
            logger.info(f"Added {len(samples)} samples from {dataset_name}")
        
        # Cache the knowledge base
        if self.cache_manager:
            metadata = {
                'num_datasets': len(datasets_dict),
                'max_samples_per_dataset': max_samples_per_dataset,
                'datasets': list(datasets_dict.keys())
            }
            self.cache_manager.cache_training_data(cache_key, self.knowledge_base, metadata)
        
        return self.knowledge_base
    
    def train(self, **kwargs) -> bool:
        """Save knowledge base"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save metadata only (samples are too large)
            metadata = {
                dataset: {'count': info['count']}
                for dataset, info in self.knowledge_base.items()
            }
            
            with open(os.path.join(self.output_dir, "knowledge_base.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Knowledge base saved to {self.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def load_trained_model(self, model_path: str = None):
        """Load knowledge base"""
        if model_path is None:
            model_path = self.output_dir
        
        try:
            kb_path = os.path.join(model_path, "knowledge_base.json")
            if os.path.exists(kb_path):
                with open(kb_path, "r") as f:
                    metadata = json.load(f)
                logger.info(f"Loaded knowledge base with {len(metadata)} datasets")
                return True
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
        
        return False
