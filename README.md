# LunaAI

LunaAI is a HuggingFace datasets integration tool that provides easy access to multiple AI/ML datasets.

## Features

This project integrates the following HuggingFace datasets:

1. **google/MapTrace** - Maps and traces dataset
2. **poloclub/diffusiondb** - Large-scale text-to-image prompts and images
3. **HuggingFaceM4/WebSight** - Synthetic web pages dataset
4. **HuggingFaceVLA/community_dataset_v1** - Community contributed dataset
5. **HuggingFaceM4/FineVision** - Vision-language dataset
6. **mrmrx/CADS-dataset** - CADS dataset
7. **PleIAs/SYNTH** - Synthetic dataset
8. **wikimedia/wikipedia** - Wikipedia dataset
9. **zwhe99/DeepMath-103K** - Mathematics problems dataset
10. **nvidia/AceMath-RM-Training-Data** - NVIDIA AceMath reward model training data
11. **HuggingFaceTB/smoltalk2** - Conversational dataset

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application

LunaAI now includes a graphical user interface for easy interaction with datasets:

```bash
python luna_gui.py
```

The GUI provides:
- **Dataset Selection**: Choose from all available HuggingFace datasets
- **Load Datasets**: Load individual datasets or all at once
- **Query Interface**: Search through loaded datasets
- **Results Display**: View dataset contents and search results

### Basic Usage (Programmatic)

```python
from huggingface_datasets import HuggingFaceDatasetLoader

# Initialize the loader
loader = HuggingFaceDatasetLoader()

# Load a specific dataset (with streaming for large datasets)
websight = loader.load_websight(streaming=True)
deepmath = loader.load_deepmath(streaming=True)

# List available datasets
print(loader.list_available_datasets())
```

### Loading All Datasets

```python
# Load all datasets with streaming enabled
loader.load_all_datasets(streaming=True)

# Get loaded datasets
datasets = loader.get_loaded_datasets()
print(f"Loaded {len(datasets)} datasets")
```

### Running the Demo

```bash
python huggingface_datasets.py
```

### Running Examples

```bash
# Run all examples
python examples.py

# Or use individual examples in your code
from examples import example_load_single_dataset, example_load_multiple_datasets
```

## Dataset Details

### Vision & Image Datasets
- **DiffusionDB**: Text-to-image generation prompts and images
- **WebSight**: Synthetic web pages for vision-language tasks
- **FineVision**: High-quality vision-language dataset

### Text & Language Datasets
- **Wikipedia**: Comprehensive encyclopedia content
- **SmolTalk2**: Conversational AI dataset
- **SYNTH**: Synthetic text generation dataset

### Math & Reasoning Datasets
- **DeepMath-103K**: Mathematical problems and solutions
- **AceMath-RM-Training-Data**: Reward model training data for math

### Specialized Datasets
- **MapTrace**: Geographic mapping data
- **Community Dataset v1**: Multi-domain community contributions
- **CADS**: Specialized domain dataset

## API Reference

### HuggingFaceDatasetLoader

Main class for loading and managing datasets.

#### Methods

- `load_maptrace(split='train', streaming=False)` - Load Google MapTrace dataset
- `load_diffusiondb(split='train', streaming=False, subset='2m_random_1k')` - Load DiffusionDB
- `load_websight(split='train', streaming=False)` - Load WebSight dataset
- `load_community_dataset(split='train', streaming=False)` - Load Community Dataset v1
- `load_finevision(split='train', streaming=False)` - Load FineVision dataset
- `load_cads(split='train', streaming=False)` - Load CADS dataset
- `load_synth(split='train', streaming=False)` - Load SYNTH dataset
- `load_wikipedia(split='train', streaming=False, language='en', date='20231101')` - Load Wikipedia
- `load_deepmath(split='train', streaming=False)` - Load DeepMath-103K dataset
- `load_acemath(split='train', streaming=False)` - Load AceMath-RM-Training-Data
- `load_smoltalk(split='train', streaming=False)` - Load SmolTalk2 dataset
- `load_all_datasets(streaming=True)` - Load all available datasets
- `list_available_datasets()` - Get list of available dataset keys
- `get_loaded_datasets()` - Get dictionary of loaded datasets
- `get_dataset_info(dataset_key)` - Get information about a specific dataset

## Requirements

- Python 3.7+
- datasets>=2.14.0
- huggingface_hub>=0.16.0

## License

See LICENSE file for details.
