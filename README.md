# Google Gemma 3N Hackathon Project

This project is built for the [Google Gemma 3N Hackathon](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/), leveraging the Gemma family of models for advanced language processing tasks.

## Project Structure

```
google-gemma-3n-hackathon/
├── src/
│   ├── models/         # Model implementations
│   ├── data/          # Data loading and preprocessing
│   └── utils/         # Utility functions
├── notebooks/         # Jupyter notebooks for experimentation
├── main.py           # Main entry point
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── requirements.txt  # Python dependencies
└── config.yaml      # Configuration file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face authentication (if needed):
```bash
huggingface-cli login
```

## Usage

### Training

```bash
python train.py --config config.yaml --data path/to/training_data.json --output outputs/my_model
```

### Evaluation

```bash
python evaluate.py --model outputs/my_model/final_model --data path/to/test_data.json --output outputs/evaluation
```

### Inference

```bash
python main.py --task predict --model_path outputs/my_model/final_model --data_path path/to/input.txt
```

## Configuration

Edit `config.yaml` to customize:
- Model selection (gemma-2b or gemma-7b)
- Training hyperparameters
- Generation parameters
- Hardware settings

## Data Format

The data loader supports multiple formats:
- JSON: `{"text": "...", "target": "..."}`
- CSV: Columns for input and output text
- JSONL: One JSON object per line
- HuggingFace datasets

## Features

- Easy-to-use interface for Gemma models
- Support for fine-tuning on custom datasets
- Comprehensive evaluation metrics
- Flexible data loading from multiple formats
- Configurable training and generation parameters

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- HuggingFace account for model access

## License

This project is created for the Google Gemma 3N Hackathon competition.