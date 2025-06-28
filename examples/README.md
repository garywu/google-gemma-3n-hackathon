# Gemma Model Examples

This directory contains example datasets and usage instructions for the Google Gemma 3N Hackathon project.

## Quick Start

### 1. Set up the environment

```bash
# Create virtual environment with UV
make setup

# Activate the environment
source .venv/bin/activate
```

### 2. Interactive Chat Mode

Start an interactive chat session with Gemma:

```bash
python main.py chat
```

This will load the default Gemma-2B model and allow you to have a conversation.

### 3. Batch Prediction

Generate responses for multiple prompts:

```bash
python main.py predict --data-path examples/data/sample_prompts.json
```

Results will be saved to `outputs/predictions.json`.

### 4. Training Example

Fine-tune Gemma on your own data:

```bash
# Using JSONL format
python main.py train --data-path examples/data/sample_training_data.jsonl

# Using instruction tuning format
python main.py train --data-path examples/data/instruction_tuning_sample.json
```

### 5. Model Evaluation

Evaluate a trained model:

```bash
python main.py evaluate \
    --model-path outputs/checkpoints \
    --data-path examples/data/sample_training_data.jsonl
```

## Data Formats

### 1. Simple Text (JSONL)

Each line is a JSON object with a "text" field:

```jsonl
{"text": "This is a training example."}
{"text": "Another training example."}
```

### 2. Instruction Tuning (JSON)

For instruction-following tasks:

```json
{
  "data": [
    {
      "input": "Write a poem about AI",
      "output": "Silicon dreams in circuits deep..."
    }
  ]
}
```

### 3. Prompts for Generation (JSON)

```json
{
  "prompts": [
    "Generate a story about...",
    "Explain the concept of..."
  ]
}
```

### 4. CSV Format

Create a CSV file with columns for input/output:

```csv
text,label
"Input text here","Expected output"
```

## Command Line Options

### Training Options

```bash
python main.py train \
    --data-path <path> \
    --model-name gemma-7b \      # Use larger model
    --epochs 5 \                 # Number of epochs
    --batch-size 8 \             # Batch size
    --learning-rate 1e-5 \       # Learning rate
    --output-dir my_model        # Save location
```

### Generation Options

```bash
python main.py predict \
    --model-path <checkpoint> \  # Use fine-tuned model
    --max-tokens 512 \           # Maximum generation length
    --temperature 0.8 \          # Creativity (0-1)
    --top-p 0.95                 # Nucleus sampling
```

## Advanced Usage

### Using Different Model Variants

```bash
# Instruction-tuned model for better chat
python main.py chat --model-name gemma-2b-it

# Larger model for complex tasks
python main.py predict --model-name gemma-7b --data-path prompts.json
```

### Custom Configuration

Edit `config.yaml` for persistent settings:

```yaml
model:
  name: gemma-2b-it
  device: cuda
  load_in_8bit: true  # For memory efficiency

generation:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9

training:
  num_epochs: 3
  per_device_batch_size: 4
  learning_rate: 2e-5
```

### Memory-Efficient Training

For limited GPU memory:

```bash
python main.py train \
    --data-path data.jsonl \
    --model-name gemma-2b \
    --batch-size 1 \
    --gradient-accumulation-steps 8
```

## Tips and Best Practices

1. **Start Small**: Begin with gemma-2b before moving to gemma-7b
2. **Data Quality**: Ensure your training data is clean and well-formatted
3. **Monitor Training**: Check the logs in `outputs/` for training progress
4. **Save Checkpoints**: Models are automatically saved during training
5. **Experiment**: Try different temperatures and sampling parameters

## Troubleshooting

### Out of Memory

- Use `--load-in-8bit` flag
- Reduce batch size
- Use gradient accumulation
- Switch to smaller model

### Slow Generation

- Reduce `max_tokens`
- Use smaller model
- Enable Flash Attention (default)

### Poor Quality Output

- Train for more epochs
- Use larger/instruction-tuned model
- Adjust temperature and top_p
- Improve training data quality

## Example Notebooks

Check the `notebooks/` directory for Jupyter notebooks demonstrating:
- Data preparation
- Model fine-tuning
- Custom evaluation metrics
- Advanced generation techniques