"""
HuggingFace Datasets Integration for LunaAI
This module provides functionality to load and process multiple HuggingFace datasets.
"""

from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HuggingFaceDatasetLoader:
    """
    A class to manage loading and processing of multiple HuggingFace datasets.
    
    Supported datasets:
    - google/MapTrace: Maps and traces dataset
    - poloclub/diffusiondb: Diffusion models dataset
    - HuggingFaceM4/WebSight: Web content dataset
    - HuggingFaceVLA/community_dataset_v1: Community contributed dataset
    - HuggingFaceM4/FineVision: Vision-language dataset
    - mrmrx/CADS-dataset: CADS dataset
    - PleIAs/SYNTH: Synthetic dataset
    - wikimedia/wikipedia: Wikipedia dataset
    - zwhe99/DeepMath-103K: Mathematics problems dataset
    - nvidia/AceMath-RM-Training-Data: NVIDIA AceMath reward model training data
    - HuggingFaceTB/smoltalk2: Conversational dataset
    - Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b: Superior reasoning dataset
    - nvidia/AudioSkills: Audio skills dataset
    - google/mobile-actions: Mobile actions dataset
    """
    
    def __init__(self):
        self.datasets = {}
        self.dataset_configs = {
            'maptrace': {
                'name': 'google/MapTrace',
                'description': 'Google MapTrace dataset'
            },
            'diffusiondb': {
                'name': 'poloclub/diffusiondb',
                'description': 'DiffusionDB - Large-scale text-to-image prompts and images'
            },
            'websight': {
                'name': 'HuggingFaceM4/WebSight',
                'description': 'WebSight - Synthetic web pages dataset'
            },
            'community_dataset': {
                'name': 'HuggingFaceVLA/community_dataset_v1',
                'description': 'HuggingFace VLA Community Dataset v1'
            },
            'finevision': {
                'name': 'HuggingFaceM4/FineVision',
                'description': 'FineVision - Vision-language dataset'
            },
            'cads': {
                'name': 'mrmrx/CADS-dataset',
                'description': 'CADS Dataset'
            },
            'synth': {
                'name': 'PleIAs/SYNTH',
                'description': 'SYNTH - Synthetic dataset by PleIAs'
            },
            'wikipedia': {
                'name': 'wikimedia/wikipedia',
                'description': 'Wikipedia dataset by Wikimedia'
            },
            'deepmath': {
                'name': 'zwhe99/DeepMath-103K',
                'description': 'DeepMath-103K - Mathematics problem dataset'
            },
            'acemath': {
                'name': 'nvidia/AceMath-RM-Training-Data',
                'description': 'AceMath RM Training Data by NVIDIA'
            },
            'smoltalk': {
                'name': 'HuggingFaceTB/smoltalk2',
                'description': 'SmolTalk2 - Conversational dataset'
            },
            'superior_reasoning': {
                'name': 'Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b',
                'description': 'Superior Reasoning SFT - Advanced reasoning dataset by Alibaba'
            },
            'audioskills': {
                'name': 'nvidia/AudioSkills',
                'description': 'AudioSkills - Audio processing dataset by NVIDIA'
            },
            'mobile_actions': {
                'name': 'google/mobile-actions',
                'description': 'Mobile Actions - Mobile interaction dataset by Google'
            }
        }
    
    def load_maptrace(self, split='train', streaming=False):
        """
        Load the Google MapTrace dataset.
        
        Args:
            split (str): Dataset split to load (e.g., 'train', 'test', 'validation')
            streaming (bool): Whether to stream the dataset instead of downloading it entirely
            
        Returns:
            Dataset: The loaded MapTrace dataset
        """
        try:
            logger.info(f"Loading MapTrace dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('google/MapTrace', split=split, streaming=streaming)
            self.datasets['maptrace'] = dataset
            logger.info("MapTrace dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading MapTrace dataset: {e}")
            raise
    
    def load_diffusiondb(self, split='train', streaming=False, subset='2m_random_1k'):
        """
        Load the DiffusionDB dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            subset (str): Subset configuration (e.g., '2m_random_1k', '2m_first_1k')
            
        Returns:
            Dataset: The loaded DiffusionDB dataset
        """
        try:
            logger.info(f"Loading DiffusionDB dataset (split: {split}, subset: {subset}, streaming: {streaming})")
            dataset = load_dataset('poloclub/diffusiondb', subset, split=split, streaming=streaming)
            self.datasets['diffusiondb'] = dataset
            logger.info("DiffusionDB dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading DiffusionDB dataset: {e}")
            raise
    
    def load_websight(self, split='train', streaming=False):
        """
        Load the HuggingFaceM4 WebSight dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded WebSight dataset
        """
        try:
            logger.info(f"Loading WebSight dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('HuggingFaceM4/WebSight', split=split, streaming=streaming)
            self.datasets['websight'] = dataset
            logger.info("WebSight dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading WebSight dataset: {e}")
            raise
    
    def load_community_dataset(self, split='train', streaming=False):
        """
        Load the HuggingFaceVLA community_dataset_v1.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded community dataset
        """
        try:
            logger.info(f"Loading Community Dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('HuggingFaceVLA/community_dataset_v1', split=split, streaming=streaming)
            self.datasets['community_dataset'] = dataset
            logger.info("Community Dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading Community Dataset: {e}")
            raise
    
    def load_finevision(self, split='train', streaming=False):
        """
        Load the HuggingFaceM4 FineVision dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded FineVision dataset
        """
        try:
            logger.info(f"Loading FineVision dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('HuggingFaceM4/FineVision', split=split, streaming=streaming)
            self.datasets['finevision'] = dataset
            logger.info("FineVision dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading FineVision dataset: {e}")
            raise
    
    def load_cads(self, split='train', streaming=False):
        """
        Load the CADS dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded CADS dataset
        """
        try:
            logger.info(f"Loading CADS dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('mrmrx/CADS-dataset', split=split, streaming=streaming)
            self.datasets['cads'] = dataset
            logger.info("CADS dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading CADS dataset: {e}")
            raise
    
    def load_synth(self, split='train', streaming=False):
        """
        Load the PleIAs SYNTH dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded SYNTH dataset
        """
        try:
            logger.info(f"Loading SYNTH dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('PleIAs/SYNTH', split=split, streaming=streaming)
            self.datasets['synth'] = dataset
            logger.info("SYNTH dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading SYNTH dataset: {e}")
            raise
    
    def load_wikipedia(self, split='train', streaming=False, language='en', date='20231101'):
        """
        Load the Wikimedia Wikipedia dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            language (str): Wikipedia language code (e.g., 'en', 'es', 'fr')
            date (str): Wikipedia dump date (format: YYYYMMDD)
            
        Returns:
            Dataset: The loaded Wikipedia dataset
        """
        try:
            logger.info(f"Loading Wikipedia dataset (language: {language}, date: {date}, split: {split}, streaming: {streaming})")
            dataset = load_dataset('wikimedia/wikipedia', f'{date}.{language}', split=split, streaming=streaming)
            self.datasets['wikipedia'] = dataset
            logger.info("Wikipedia dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading Wikipedia dataset: {e}")
            raise
    
    def load_deepmath(self, split='train', streaming=False):
        """
        Load the DeepMath-103K dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded DeepMath dataset
        """
        try:
            logger.info(f"Loading DeepMath-103K dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('zwhe99/DeepMath-103K', split=split, streaming=streaming)
            self.datasets['deepmath'] = dataset
            logger.info("DeepMath-103K dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading DeepMath-103K dataset: {e}")
            raise
    
    def load_acemath(self, split='train', streaming=False):
        """
        Load the NVIDIA AceMath-RM-Training-Data dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded AceMath dataset
        """
        try:
            logger.info(f"Loading AceMath-RM-Training-Data dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('nvidia/AceMath-RM-Training-Data', split=split, streaming=streaming)
            self.datasets['acemath'] = dataset
            logger.info("AceMath-RM-Training-Data dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading AceMath-RM-Training-Data dataset: {e}")
            raise
    
    def load_smoltalk(self, split='train', streaming=False):
        """
        Load the HuggingFaceTB smoltalk2 dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded SmolTalk2 dataset
        """
        try:
            logger.info(f"Loading SmolTalk2 dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('HuggingFaceTB/smoltalk2', split=split, streaming=streaming)
            self.datasets['smoltalk'] = dataset
            logger.info("SmolTalk2 dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading SmolTalk2 dataset: {e}")
            raise
    
    def load_superior_reasoning(self, split='train', streaming=False):
        """
        Load the Alibaba-Apsara Superior-Reasoning-SFT dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded Superior Reasoning dataset
        """
        try:
            logger.info(f"Loading Superior Reasoning dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b', split=split, streaming=streaming)
            self.datasets['superior_reasoning'] = dataset
            logger.info("Superior Reasoning dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading Superior Reasoning dataset: {e}")
            raise
    
    def load_audioskills(self, split='train', streaming=False):
        """
        Load the NVIDIA AudioSkills dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded AudioSkills dataset
        """
        try:
            logger.info(f"Loading AudioSkills dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('nvidia/AudioSkills', split=split, streaming=streaming)
            self.datasets['audioskills'] = dataset
            logger.info("AudioSkills dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading AudioSkills dataset: {e}")
            raise
    
    def load_mobile_actions(self, split='train', streaming=False):
        """
        Load the Google mobile-actions dataset.
        
        Args:
            split (str): Dataset split to load
            streaming (bool): Whether to stream the dataset
            
        Returns:
            Dataset: The loaded Mobile Actions dataset
        """
        try:
            logger.info(f"Loading Mobile Actions dataset (split: {split}, streaming: {streaming})")
            dataset = load_dataset('google/mobile-actions', split=split, streaming=streaming)
            self.datasets['mobile_actions'] = dataset
            logger.info("Mobile Actions dataset loaded successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error loading Mobile Actions dataset: {e}")
            raise
    
    def load_all_datasets(self, streaming=True):
        """
        Load all supported datasets.
        
        Args:
            streaming (bool): Whether to stream datasets (recommended for large datasets)
            
        Returns:
            dict: Dictionary containing all loaded datasets
        """
        logger.info("Loading all datasets...")
        
        try:
            self.load_maptrace(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load MapTrace: {e}")
        
        try:
            self.load_diffusiondb(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load DiffusionDB: {e}")
        
        try:
            self.load_websight(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load WebSight: {e}")
        
        try:
            self.load_community_dataset(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load Community Dataset: {e}")
        
        try:
            self.load_finevision(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load FineVision: {e}")
        
        try:
            self.load_cads(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load CADS: {e}")
        
        try:
            self.load_synth(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load SYNTH: {e}")
        
        try:
            self.load_wikipedia(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load Wikipedia: {e}")
        
        try:
            self.load_deepmath(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load DeepMath-103K: {e}")
        
        try:
            self.load_acemath(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load AceMath-RM-Training-Data: {e}")
        
        try:
            self.load_smoltalk(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load SmolTalk2: {e}")
        
        try:
            self.load_superior_reasoning(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load Superior Reasoning: {e}")
        
        try:
            self.load_audioskills(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load AudioSkills: {e}")
        
        try:
            self.load_mobile_actions(streaming=streaming)
        except Exception as e:
            logger.warning(f"Could not load Mobile Actions: {e}")
        
        logger.info(f"Loaded {len(self.datasets)} datasets successfully")
        return self.datasets
    
    def get_dataset_info(self, dataset_key):
        """
        Get information about a specific dataset.
        
        Args:
            dataset_key (str): Key of the dataset (e.g., 'maptrace', 'diffusiondb', 'websight', 
                              'community_dataset', 'finevision', 'cads', 'synth', 'wikipedia', 
                              'deepmath', 'acemath', 'smoltalk', 'superior_reasoning', 
                              'audioskills', 'mobile_actions')
            
        Returns:
            dict: Dataset information
        """
        if dataset_key in self.dataset_configs:
            return self.dataset_configs[dataset_key]
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
    
    def list_available_datasets(self):
        """
        List all available datasets.
        
        Returns:
            list: List of available dataset keys
        """
        return list(self.dataset_configs.keys())
    
    def get_loaded_datasets(self):
        """
        Get all currently loaded datasets.
        
        Returns:
            dict: Dictionary of loaded datasets
        """
        return self.datasets


def main():
    """
    Example usage of the HuggingFaceDatasetLoader.
    """
    logger.info("=== HuggingFace Datasets Integration Demo ===")
    
    # Initialize the loader
    loader = HuggingFaceDatasetLoader()
    
    # List available datasets
    logger.info(f"Available datasets: {loader.list_available_datasets()}")
    
    # Example: Load datasets with streaming enabled (recommended for large datasets)
    logger.info("\nLoading datasets with streaming enabled...")
    
    try:
        # Load MapTrace
        maptrace = loader.load_maptrace(streaming=True)
        logger.info(f"MapTrace dataset loaded: {type(maptrace)}")
    except Exception as e:
        logger.error(f"Failed to load MapTrace: {e}")
    
    try:
        # Load DiffusionDB (using a smaller subset for demo)
        diffusiondb = loader.load_diffusiondb(streaming=True, subset='2m_random_1k')
        logger.info(f"DiffusionDB dataset loaded: {type(diffusiondb)}")
    except Exception as e:
        logger.error(f"Failed to load DiffusionDB: {e}")
    
    try:
        # Load WebSight
        websight = loader.load_websight(streaming=True)
        logger.info(f"WebSight dataset loaded: {type(websight)}")
    except Exception as e:
        logger.error(f"Failed to load WebSight: {e}")
    
    try:
        # Load Community Dataset
        community = loader.load_community_dataset(streaming=True)
        logger.info(f"Community Dataset loaded: {type(community)}")
    except Exception as e:
        logger.error(f"Failed to load Community Dataset: {e}")
    
    try:
        # Load FineVision
        finevision = loader.load_finevision(streaming=True)
        logger.info(f"FineVision dataset loaded: {type(finevision)}")
    except Exception as e:
        logger.error(f"Failed to load FineVision: {e}")
    
    try:
        # Load CADS
        cads = loader.load_cads(streaming=True)
        logger.info(f"CADS dataset loaded: {type(cads)}")
    except Exception as e:
        logger.error(f"Failed to load CADS: {e}")
    
    try:
        # Load SYNTH
        synth = loader.load_synth(streaming=True)
        logger.info(f"SYNTH dataset loaded: {type(synth)}")
    except Exception as e:
        logger.error(f"Failed to load SYNTH: {e}")
    
    try:
        # Load Wikipedia (English)
        wikipedia = loader.load_wikipedia(streaming=True, language='en', date='20231101')
        logger.info(f"Wikipedia dataset loaded: {type(wikipedia)}")
    except Exception as e:
        logger.error(f"Failed to load Wikipedia: {e}")
    
    try:
        # Load DeepMath-103K
        deepmath = loader.load_deepmath(streaming=True)
        logger.info(f"DeepMath-103K dataset loaded: {type(deepmath)}")
    except Exception as e:
        logger.error(f"Failed to load DeepMath-103K: {e}")
    
    try:
        # Load AceMath-RM-Training-Data
        acemath = loader.load_acemath(streaming=True)
        logger.info(f"AceMath-RM-Training-Data dataset loaded: {type(acemath)}")
    except Exception as e:
        logger.error(f"Failed to load AceMath-RM-Training-Data: {e}")
    
    try:
        # Load SmolTalk2
        smoltalk = loader.load_smoltalk(streaming=True)
        logger.info(f"SmolTalk2 dataset loaded: {type(smoltalk)}")
    except Exception as e:
        logger.error(f"Failed to load SmolTalk2: {e}")
    
    # Show loaded datasets
    loaded = loader.get_loaded_datasets()
    logger.info(f"\nSuccessfully loaded {len(loaded)} datasets: {list(loaded.keys())}")


if __name__ == "__main__":
    main()
