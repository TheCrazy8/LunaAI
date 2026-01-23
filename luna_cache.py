"""
Luna Cache Manager
Handles caching of training data, models, and datasets across instances
"""

import logging
import os
import json
import pickle
import hashlib
from typing import Any, Optional, Dict
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LunaCacheManager:
    """
    Manages caching for Luna AI to persist across instances
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir (str): Directory for cache storage. Defaults to ~/.luna_cache
        """
        if cache_dir is None:
            # Use user's home directory
            cache_dir = os.path.expanduser("~/.luna_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different cache types
        self.models_dir = self.cache_dir / "models"
        self.datasets_dir = self.cache_dir / "datasets"
        self.training_dir = self.cache_dir / "training"
        self.metadata_dir = self.cache_dir / "metadata"
        
        for dir_path in [self.models_dir, self.datasets_dir, self.training_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"Luna Cache Manager initialized at: {self.cache_dir}")
        
        # Load cache index
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict:
        """Load the cache index"""
        index_file = self.metadata_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache index: {e}")
        
        return {
            'models': {},
            'datasets': {},
            'training_data': {},
            'created': time.time(),
            'last_updated': time.time()
        }
    
    def _save_cache_index(self):
        """Save the cache index"""
        index_file = self.metadata_dir / "cache_index.json"
        self.cache_index['last_updated'] = time.time()
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate a unique cache key from data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def cache_trained_model(self, model_name: str, model_path: str, 
                           metadata: Dict = None) -> bool:
        """
        Cache a trained model.
        
        Args:
            model_name (str): Name/identifier for the model
            model_path (str): Path to the model directory
            metadata (dict): Additional metadata about the model
            
        Returns:
            bool: True if successful
        """
        try:
            target_dir = self.models_dir / model_name
            
            # Copy model files to cache
            import shutil
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(model_path, target_dir)
            
            # Update cache index
            self.cache_index['models'][model_name] = {
                'path': str(target_dir),
                'cached_at': time.time(),
                'metadata': metadata or {}
            }
            self._save_cache_index()
            
            logger.info(f"Model '{model_name}' cached successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error caching model: {e}")
            return False
    
    def get_cached_model(self, model_name: str) -> Optional[str]:
        """
        Get path to a cached model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            str: Path to cached model or None if not found
        """
        if model_name in self.cache_index['models']:
            model_info = self.cache_index['models'][model_name]
            model_path = Path(model_info['path'])
            
            if model_path.exists():
                logger.info(f"Found cached model: {model_name}")
                return str(model_path)
            else:
                logger.warning(f"Cached model path not found: {model_path}")
                # Clean up stale entry
                del self.cache_index['models'][model_name]
                self._save_cache_index()
        
        return None
    
    def cache_training_data(self, data_name: str, data: Any, 
                           metadata: Dict = None) -> bool:
        """
        Cache training data.
        
        Args:
            data_name (str): Identifier for the training data
            data (Any): Training data to cache (will be pickled)
            metadata (dict): Metadata about the training data
            
        Returns:
            bool: True if successful
        """
        try:
            cache_file = self.training_dir / f"{data_name}.pkl"
            
            # Pickle the data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update cache index
            self.cache_index['training_data'][data_name] = {
                'path': str(cache_file),
                'cached_at': time.time(),
                'metadata': metadata or {}
            }
            self._save_cache_index()
            
            logger.info(f"Training data '{data_name}' cached successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error caching training data: {e}")
            return False
    
    def get_cached_training_data(self, data_name: str) -> Optional[Any]:
        """
        Get cached training data.
        
        Args:
            data_name (str): Name of the cached training data
            
        Returns:
            Any: Cached data or None if not found
        """
        if data_name in self.cache_index['training_data']:
            data_info = self.cache_index['training_data'][data_name]
            cache_file = Path(data_info['path'])
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Loaded cached training data: {data_name}")
                    return data
                except Exception as e:
                    logger.error(f"Error loading cached data: {e}")
            else:
                logger.warning(f"Cached data file not found: {cache_file}")
                # Clean up stale entry
                del self.cache_index['training_data'][data_name]
                self._save_cache_index()
        
        return None
    
    def cache_dataset_samples(self, dataset_name: str, samples: list, 
                             metadata: Dict = None) -> bool:
        """
        Cache dataset samples for quick loading.
        
        Args:
            dataset_name (str): Name of the dataset
            samples (list): List of dataset samples
            metadata (dict): Metadata about the dataset
            
        Returns:
            bool: True if successful
        """
        try:
            cache_file = self.datasets_dir / f"{dataset_name}.pkl"
            
            data = {
                'samples': samples,
                'metadata': metadata or {},
                'cached_at': time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update cache index
            self.cache_index['datasets'][dataset_name] = {
                'path': str(cache_file),
                'sample_count': len(samples),
                'cached_at': time.time(),
                'metadata': metadata or {}
            }
            self._save_cache_index()
            
            logger.info(f"Dataset '{dataset_name}' samples cached ({len(samples)} samples)")
            return True
            
        except Exception as e:
            logger.error(f"Error caching dataset samples: {e}")
            return False
    
    def get_cached_dataset_samples(self, dataset_name: str) -> Optional[list]:
        """
        Get cached dataset samples.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            list: List of samples or None if not found
        """
        if dataset_name in self.cache_index['datasets']:
            dataset_info = self.cache_index['datasets'][dataset_name]
            cache_file = Path(dataset_info['path'])
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Loaded cached dataset: {dataset_name} ({len(data['samples'])} samples)")
                    return data['samples']
                except Exception as e:
                    logger.error(f"Error loading cached dataset: {e}")
            else:
                logger.warning(f"Cached dataset file not found: {cache_file}")
                # Clean up stale entry
                del self.cache_index['datasets'][dataset_name]
                self._save_cache_index()
        
        return None
    
    def list_cached_models(self) -> Dict[str, Dict]:
        """List all cached models"""
        return self.cache_index['models'].copy()
    
    def list_cached_training_data(self) -> Dict[str, Dict]:
        """List all cached training data"""
        return self.cache_index['training_data'].copy()
    
    def list_cached_datasets(self) -> Dict[str, Dict]:
        """List all cached datasets"""
        return self.cache_index['datasets'].copy()
    
    def clear_cache(self, cache_type: str = None) -> bool:
        """
        Clear cache.
        
        Args:
            cache_type (str): Type to clear ('models', 'datasets', 'training_data', or None for all)
            
        Returns:
            bool: True if successful
        """
        try:
            import shutil
            
            if cache_type is None or cache_type == 'models':
                if self.models_dir.exists():
                    shutil.rmtree(self.models_dir)
                    self.models_dir.mkdir()
                self.cache_index['models'] = {}
                logger.info("Cleared models cache")
            
            if cache_type is None or cache_type == 'datasets':
                if self.datasets_dir.exists():
                    shutil.rmtree(self.datasets_dir)
                    self.datasets_dir.mkdir()
                self.cache_index['datasets'] = {}
                logger.info("Cleared datasets cache")
            
            if cache_type is None or cache_type == 'training_data':
                if self.training_dir.exists():
                    shutil.rmtree(self.training_dir)
                    self.training_dir.mkdir()
                self.cache_index['training_data'] = {}
                logger.info("Cleared training data cache")
            
            self._save_cache_index()
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_size(self) -> Dict[str, int]:
        """Get size of cache in bytes"""
        sizes = {
            'models': 0,
            'datasets': 0,
            'training': 0,
            'total': 0
        }
        
        try:
            for path in self.models_dir.rglob('*'):
                if path.is_file():
                    sizes['models'] += path.stat().st_size
            
            for path in self.datasets_dir.rglob('*'):
                if path.is_file():
                    sizes['datasets'] += path.stat().st_size
            
            for path in self.training_dir.rglob('*'):
                if path.is_file():
                    sizes['training'] += path.stat().st_size
            
            sizes['total'] = sizes['models'] + sizes['datasets'] + sizes['training']
            
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
        
        return sizes
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        sizes = self.get_cache_size()
        
        return {
            'cache_dir': str(self.cache_dir),
            'models_count': len(self.cache_index['models']),
            'datasets_count': len(self.cache_index['datasets']),
            'training_data_count': len(self.cache_index['training_data']),
            'total_size_mb': sizes['total'] / (1024 * 1024),
            'models_size_mb': sizes['models'] / (1024 * 1024),
            'datasets_size_mb': sizes['datasets'] / (1024 * 1024),
            'training_size_mb': sizes['training'] / (1024 * 1024),
            'created': self.cache_index.get('created', 0),
            'last_updated': self.cache_index.get('last_updated', 0)
        }
