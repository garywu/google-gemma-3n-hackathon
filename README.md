# Google Gemma 3N Hackathon Project 🚀

A **production-ready framework** for the [Google Gemma 3N Hackathon](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/) that transforms Google's Gemma models into reliable AI systems with:

- ⚡ **2-5x faster training** with Unsloth optimization
- 🎯 **100% structured outputs** with Pydantic + Instructor
- 🧠 **Self-optimizing prompts** with DSPy
- 🔍 **Comprehensive evaluation** with Inspect AI
- 🌐 **Production API** with FastAPI

## 🏆 Key Features

### 1. **Structured Outputs** - Never Parse JSON Again!
```python
# Always get valid, typed responses
analysis = pipeline.instructor.analyze_code(code)
print(f"Score: {analysis.score}/10")  # Guaranteed float!
print(f"Issues: {analysis.improvements}")  # Always a list!
```

### 2. **Lightning-Fast Training** - Unsloth Integration
```python
# Train 2-5x faster with 70% less memory
model = GemmaModelUnsloth("gemma-2b-4bit")
model.train(dataset)  # Optimized with LoRA + 4-bit
```

### 3. **Self-Optimizing** - DSPy Magic
```python
# No more prompt engineering!
optimized = dspy.compile(generator, examples)
# Automatically finds best prompts for YOUR data
```

### 4. **Production API** - Ready to Deploy
```python
# Reliable API with guaranteed responses
POST /analyze-code
POST /generate-tests
POST /debug-code
# All return validated JSON every time!
```

## 🚀 Quick Start

```bash
# 1. Create environment with UV (fast!)
make setup

# 2. Activate environment
source .venv/bin/activate.fish  # or .venv/bin/activate for bash

# 3. Run interactive chat
python main.py chat

# 4. Start API server
python api_server.py
```

## Project Structure

```
google-gemma-3n-hackathon/
├── src/
│   ├── models/         # Model implementations
│   │   ├── gemma_model.py         # Base Gemma wrapper
│   │   └── gemma_model_unsloth.py # Optimized with Unsloth
│   ├── data/          # Data loading and preprocessing
│   ├── structured/    # Pydantic + Instructor + DSPy
│   └── evaluation/    # Inspect AI integration
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