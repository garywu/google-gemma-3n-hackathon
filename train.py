import argparse
import logging
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer

from src.models.gemma_model import GemmaModel
from src.data.data_loader import DataLoader
from src.utils.config import load_config, save_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(config_path: str, data_path: str, output_dir: str):
    """Train Gemma model"""
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config to output dir
    save_config(config, output_dir / 'config.yaml')
    
    # Initialize model
    model = GemmaModel(config)
    model.load_model()
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # Load and preprocess data
    logger.info(f"Loading training data from {data_path}")
    dataset = data_loader.load_data(data_path)
    
    # Split into train/eval if needed
    if 'validation' in dataset:
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
    else:
        # Create train/eval split
        split_dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    
    # Preprocess datasets
    train_dataset = data_loader.preprocess_data(train_dataset, model.tokenizer)
    eval_dataset = data_loader.preprocess_data(eval_dataset, model.tokenizer)
    
    # Create dataloaders
    train_dataloader = data_loader.get_dataloader(train_dataset, shuffle=True)
    eval_dataloader = data_loader.get_dataloader(eval_dataset, shuffle=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get('epochs', 3),
        per_device_train_batch_size=config.get('batch_size', 8),
        per_device_eval_batch_size=config.get('batch_size', 8),
        warmup_steps=config.get('warmup_steps', 500),
        weight_decay=config.get('weight_decay', 0.01),
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = output_dir / 'final_model'
    model.save_model(str(final_model_path))
    logger.info(f"Training completed. Model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Gemma model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--output', type=str, default='outputs/training',
                        help='Output directory for model and logs')
    
    args = parser.parse_args()
    
    train(args.config, args.data, args.output)


if __name__ == '__main__':
    main()