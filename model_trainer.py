"""
AI Model Trainer for LunaAI
Trains a conversational AI model using HuggingFace datasets.
"""

import os
import json
import logging
from typing import Optional, Dict, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, concatenate_datasets
from huggingface_datasets import HuggingFaceDatasetLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LunaAITrainer:
    """
    Trainer class for LunaAI conversational model.
    Uses multiple datasets to train a chat-capable model.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        output_dir: str = "./luna_model",
        max_length: int = 512,
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save the trained model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.dataset_loader = HuggingFaceDatasetLoader()
        
        logger.info(f"Initialized LunaAI Trainer with base model: {model_name}")
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_training_data(
        self,
        num_samples: int = 10000,
        use_smoltalk: bool = True,
        use_acemath: bool = False,
    ) -> Dataset:
        """
        Prepare training data from HuggingFace datasets.
        
        Args:
            num_samples: Number of samples to use for training
            use_smoltalk: Whether to use SmolTalk2 dataset
            use_acemath: Whether to use AceMath dataset
            
        Returns:
            Combined dataset ready for training
        """
        logger.info("Preparing training data...")
        datasets_to_combine = []
        
        if use_smoltalk:
            try:
                logger.info("Loading SmolTalk2 dataset...")
                smoltalk = self.dataset_loader.load_smoltalk(streaming=True)
                
                # Convert streaming dataset to list of samples
                samples = []
                for i, example in enumerate(smoltalk):
                    if i >= num_samples:
                        break
                    
                    # Format conversation for training
                    if 'messages' in example:
                        text = self._format_conversation(example['messages'])
                        samples.append({'text': text})
                    elif 'prompt' in example and 'response' in example:
                        text = f"User: {example['prompt']}\nAssistant: {example['response']}"
                        samples.append({'text': text})
                
                if samples:
                    datasets_to_combine.append(Dataset.from_list(samples))
                    logger.info(f"Added {len(samples)} samples from SmolTalk2")
            except Exception as e:
                logger.error(f"Failed to load SmolTalk2: {e}")
        
        if use_acemath:
            try:
                logger.info("Loading AceMath dataset...")
                acemath = self.dataset_loader.load_acemath(streaming=True)
                
                samples = []
                for i, example in enumerate(acemath):
                    if i >= num_samples // 2:  # Use fewer math samples
                        break
                    
                    # Format math problem-solution pairs
                    if 'problem' in example and 'solution' in example:
                        text = f"User: {example['problem']}\nAssistant: {example['solution']}"
                        samples.append({'text': text})
                
                if samples:
                    datasets_to_combine.append(Dataset.from_list(samples))
                    logger.info(f"Added {len(samples)} samples from AceMath")
            except Exception as e:
                logger.error(f"Failed to load AceMath: {e}")
        
        if not datasets_to_combine:
            raise ValueError("No datasets were loaded successfully")
        
        # Combine all datasets
        combined_dataset = concatenate_datasets(datasets_to_combine)
        logger.info(f"Total training samples: {len(combined_dataset)}")
        
        return combined_dataset
    
    def _format_conversation(self, messages) -> str:
        """
        Format conversation messages into a training text.
        
        Args:
            messages: List of message dictionaries or string
            
        Returns:
            Formatted conversation string
        """
        if isinstance(messages, list):
            formatted = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'user':
                        formatted.append(f"User: {content}")
                    elif role == 'assistant':
                        formatted.append(f"Assistant: {content}")
            return "\n".join(formatted)
        elif isinstance(messages, str):
            return messages
        else:
            return str(messages)
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize the dataset for training.
        
        Args:
            dataset: Dataset to tokenize
            
        Returns:
            Tokenized dataset
        """
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            outputs = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors=None,
            )
            outputs['labels'] = outputs['input_ids'].copy()
            return outputs
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        logger.info("Dataset tokenized successfully")
        return tokenized
    
    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        save_steps: int = 500,
    ):
        """
        Train the model on the prepared dataset.
        
        Args:
            dataset: Tokenized dataset for training
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
        """
        logger.info("Starting training...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=100,
            logging_dir=f"{self.output_dir}/logs",
            report_to="none",
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Training in progress...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training config
        config = {
            'base_model': self.model_name,
            'max_length': self.max_length,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        }
        
        with open(f"{self.output_dir}/training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Training completed successfully!")
    
    def train_model(
        self,
        num_samples: int = 10000,
        num_epochs: int = 3,
        batch_size: int = 4,
    ):
        """
        Complete training pipeline: load model, prepare data, and train.
        
        Args:
            num_samples: Number of samples for training
            num_epochs: Number of training epochs
            batch_size: Training batch size
        """
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare training data
        dataset = self.prepare_training_data(
            num_samples=num_samples,
            use_smoltalk=True,
            use_acemath=False,
        )
        
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Train
        self.train(
            dataset=tokenized_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )


def main():
    """Main training script."""
    logger.info("="*60)
    logger.info("LunaAI Model Training")
    logger.info("="*60)
    
    # Initialize trainer
    trainer = LunaAITrainer(
        model_name="microsoft/DialoGPT-small",
        output_dir="./luna_model",
    )
    
    # Train model
    trainer.train_model(
        num_samples=5000,  # Use 5000 samples for faster training
        num_epochs=2,  # 2 epochs for demo
        batch_size=4,
    )
    
    logger.info("Training completed! Model saved to ./luna_model")


if __name__ == "__main__":
    main()
