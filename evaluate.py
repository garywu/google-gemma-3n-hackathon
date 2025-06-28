import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np

from src.models.gemma_model import GemmaModel
from src.data.data_loader import DataLoader
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)
    
    # Calculate averages
    metrics = {}
    for key, values in rouge_scores.items():
        metrics[f'{key}_f1'] = np.mean(values)
    
    return metrics


def evaluate(
    model_path: str,
    data_path: str,
    output_dir: str,
    config_path: str = None,
    max_samples: int = None
):
    """Evaluate Gemma model"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        # Try to load from model directory
        model_config_path = Path(model_path) / 'config.yaml'
        if model_config_path.exists():
            config = load_config(str(model_config_path))
        else:
            config = {}
    
    # Initialize model
    model = GemmaModel(config)
    model.load_model(checkpoint_path=model_path)
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load evaluation data
    logger.info(f"Loading evaluation data from {data_path}")
    dataset = data_loader.load_data(data_path)
    
    # Use test split if available
    if 'test' in dataset:
        eval_dataset = dataset['test']
    elif 'validation' in dataset:
        eval_dataset = dataset['validation']
    else:
        eval_dataset = dataset
    
    # Limit samples if specified
    if max_samples:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    
    # Generate predictions
    logger.info(f"Generating predictions for {len(eval_dataset)} samples...")
    predictions = []
    references = []
    
    for idx, example in enumerate(tqdm(eval_dataset)):
        # Get input text
        if 'input' in example:
            input_text = example['input']
        elif 'text' in example:
            input_text = example['text']
        else:
            input_text = example.get('prompt', '')
        
        # Get reference output
        if 'output' in example:
            reference = example['output']
        elif 'target' in example:
            reference = example['target']
        else:
            reference = example.get('response', '')
        
        # Generate prediction
        prediction = model.generate(
            input_text,
            max_new_tokens=config.get('max_new_tokens', 100),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 0.9)
        )
        
        predictions.append(prediction)
        references.append(reference)
        
        # Save intermediate results
        if (idx + 1) % 100 == 0:
            intermediate_results = {
                'predictions': predictions,
                'references': references
            }
            with open(output_dir / 'intermediate_results.json', 'w') as f:
                json.dump(intermediate_results, f, indent=2)
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(predictions, references)
    
    # Save results
    results = {
        'metrics': metrics,
        'num_samples': len(predictions),
        'model_path': model_path,
        'data_path': data_path,
        'predictions': predictions[:10],  # Save first 10 for inspection
        'references': references[:10]
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print metrics
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gemma model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to evaluation data')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                        help='Output directory for results')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--max-samples', type=int,
                        help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    evaluate(
        args.model,
        args.data,
        args.output,
        args.config,
        args.max_samples
    )


if __name__ == '__main__':
    main()