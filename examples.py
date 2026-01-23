"""
Example usage scripts for HuggingFace datasets integration.
"""

from huggingface_datasets import HuggingFaceDatasetLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_load_single_dataset():
    """Example: Load a single dataset"""
    logger.info("=== Example: Loading a single dataset ===")
    
    loader = HuggingFaceDatasetLoader()
    
    # Load SmolTalk2 dataset with streaming
    dataset = loader.load_smoltalk(streaming=True)
    
    # Print first few examples
    logger.info("First 3 examples from SmolTalk2:")
    for i, example in enumerate(dataset):
        if i >= 3:
            break
        logger.info(f"Example {i+1}: {example}")


def example_load_multiple_datasets():
    """Example: Load multiple specific datasets"""
    logger.info("=== Example: Loading multiple datasets ===")
    
    loader = HuggingFaceDatasetLoader()
    
    # Load math-related datasets
    try:
        deepmath = loader.load_deepmath(streaming=True)
        logger.info("DeepMath-103K loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load DeepMath: {e}")
    
    try:
        acemath = loader.load_acemath(streaming=True)
        logger.info("AceMath loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load AceMath: {e}")
    
    # Show loaded datasets
    loaded = loader.get_loaded_datasets()
    logger.info(f"Total loaded: {len(loaded)} datasets")


def example_wikipedia_with_options():
    """Example: Load Wikipedia with custom language"""
    logger.info("=== Example: Loading Wikipedia with custom options ===")
    
    loader = HuggingFaceDatasetLoader()
    
    # Load English Wikipedia
    try:
        wiki_en = loader.load_wikipedia(streaming=True, language='en', date='20231101')
        logger.info("English Wikipedia loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Wikipedia: {e}")


def example_get_dataset_info():
    """Example: Get information about available datasets"""
    logger.info("=== Example: Dataset information ===")
    
    loader = HuggingFaceDatasetLoader()
    
    # List all available datasets
    available = loader.list_available_datasets()
    logger.info(f"Available datasets: {available}")
    
    # Get info about specific datasets
    for dataset_key in ['websight', 'deepmath', 'smoltalk']:
        info = loader.get_dataset_info(dataset_key)
        logger.info(f"{dataset_key}: {info['name']} - {info['description']}")


if __name__ == "__main__":
    # Run examples
    print("\n" + "="*60)
    example_get_dataset_info()
    
    print("\n" + "="*60)
    example_load_single_dataset()
    
    print("\n" + "="*60)
    example_load_multiple_datasets()
    
    print("\n" + "="*60)
    example_wikipedia_with_options()
