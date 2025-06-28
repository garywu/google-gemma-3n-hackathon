import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from src.models.gemma_model import GemmaModel
from src.data.data_loader import DataLoader as GemmaDataLoader
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(model: GemmaModel, config: dict, data_path: str, output_dir: Path):
    """Train the Gemma model on the provided dataset."""
    # Load model
    model.load_model(config.get('model', {}).get('checkpoint_path'))
    
    # Initialize data loader
    data_loader = GemmaDataLoader(
        tokenizer=model.tokenizer,
        max_length=config.get('data', {}).get('max_length', 512)
    )
    
    # Load datasets
    logger.info(f"Loading training data from {data_path}")
    train_dataset, val_dataset = data_loader.load_from_file(
        data_path,
        input_column=config.get('data', {}).get('input_column', 'text'),
        target_column=config.get('data', {}).get('target_column'),
        validation_split=config.get('data', {}).get('validation_split', 0.1)
    )
    
    # Train model
    model.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=str(output_dir / 'checkpoints'),
        **config.get('training', {})
    )
    
    logger.info("Training completed!")


def evaluate_model(model: GemmaModel, config: dict, data_path: str, output_dir: Path):
    """Evaluate the model on a test dataset."""
    # Load model from checkpoint
    checkpoint_path = config.get('model', {}).get('checkpoint_path')
    if not checkpoint_path:
        raise ValueError("Model checkpoint path required for evaluation")
        
    model.load_model(checkpoint_path)
    
    # Initialize data loader
    data_loader = GemmaDataLoader(
        tokenizer=model.tokenizer,
        max_length=config.get('data', {}).get('max_length', 512)
    )
    
    # Load test dataset
    logger.info(f"Loading evaluation data from {data_path}")
    test_dataset = data_loader.load_from_file(
        data_path,
        input_column=config.get('data', {}).get('input_column', 'text'),
        target_column=config.get('data', {}).get('target_column'),
        validation_split=0  # No validation split for evaluation
    )
    
    # Create dataloader
    test_loader = data_loader.create_dataloader(
        test_dataset,
        batch_size=config.get('evaluation', {}).get('batch_size', 8),
        shuffle=False
    )
    
    # Evaluation metrics
    results = {
        'total_loss': 0,
        'num_samples': 0,
        'perplexity': 0
    }
    
    model.model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model.model(**batch)
            loss = outputs.loss
            
            results['total_loss'] += loss.item() * batch['input_ids'].size(0)
            results['num_samples'] += batch['input_ids'].size(0)
    
    # Calculate final metrics
    avg_loss = results['total_loss'] / results['num_samples']
    results['average_loss'] = avg_loss
    results['perplexity'] = torch.exp(torch.tensor(avg_loss)).item()
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Evaluation completed! Results saved to {results_path}")
    logger.info(f"Average Loss: {avg_loss:.4f}, Perplexity: {results['perplexity']:.4f}")


def predict_model(model: GemmaModel, config: dict, data_path: str, output_dir: Path):
    """Generate predictions using the model."""
    # Load model from checkpoint if provided
    checkpoint_path = config.get('model', {}).get('checkpoint_path')
    model.load_model(checkpoint_path)
    
    # Handle different input formats
    prompts = []
    
    if data_path:
        # Load prompts from file
        path = Path(data_path)
        if path.suffix == '.txt':
            with open(path, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
                else:
                    prompts = [str(data)]
    else:
        # Interactive mode
        logger.info("No data path provided. Entering interactive mode.")
        logger.info("Type 'quit' to exit.\n")
        
        while True:
            prompt = input("Enter prompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            # Generate response
            responses = model.generate(
                prompt,
                **config.get('generation', {})
            )
            
            print(f"\nGenerated response:\n{responses[0]}\n")
            print("-" * 50)
        
        return
    
    # Batch generation for file input
    logger.info(f"Generating predictions for {len(prompts)} prompts")
    
    all_responses = []
    generation_config = config.get('generation', {})
    
    # Process in batches
    batch_size = generation_config.get('batch_size', 1)
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]
        responses = model.generate(batch_prompts, **generation_config)
        
        for prompt, response in zip(batch_prompts, responses):
            all_responses.append({
                'prompt': prompt,
                'response': response
            })
    
    # Save results
    output_path = output_dir / 'predictions.json'
    with open(output_path, 'w') as f:
        json.dump(all_responses, f, indent=2)
        
    logger.info(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Google Gemma 3N Hackathon Project - Fine-tune and use Gemma models'
    )
    
    # Main arguments
    parser.add_argument('task', type=str, 
                        choices=['train', 'evaluate', 'predict', 'chat'],
                        help='Task to perform')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model-name', type=str,
                        help='Model name (overrides config)')
    parser.add_argument('--model-path', type=str, 
                        help='Path to saved model checkpoint')
    parser.add_argument('--data-path', type=str,
                        help='Path to input data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for outputs')
    
    # Generation arguments
    parser.add_argument('--max-tokens', type=int,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float,
                        help='Nucleus sampling threshold')
    
    # Training arguments
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.model_path:
        config['model']['checkpoint_path'] = args.model_path
    if args.max_tokens:
        config['generation']['max_new_tokens'] = args.max_tokens
    if args.temperature:
        config['generation']['temperature'] = args.temperature
    if args.top_p:
        config['generation']['top_p'] = args.top_p
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['per_device_batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Initialize model
    model_config = config.get('model', {})
    model = GemmaModel(
        model_name=model_config.get('name', 'gemma-2b'),
        device=model_config.get('device', 'auto'),
        load_in_8bit=model_config.get('load_in_8bit', False),
        use_flash_attention=model_config.get('use_flash_attention', True)
    )
    
    # Execute task
    if args.task == 'train':
        if not args.data_path:
            parser.error("--data-path required for training")
        train_model(model, config, args.data_path, output_dir)
    
    elif args.task == 'evaluate':
        if not args.data_path:
            parser.error("--data-path required for evaluation")
        evaluate_model(model, config, args.data_path, output_dir)
    
    elif args.task in ['predict', 'chat']:
        predict_model(model, config, args.data_path, output_dir)


if __name__ == '__main__':
    main()