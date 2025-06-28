#!/usr/bin/env python3
"""
Unified Training and Evaluation Pipeline for Gemma Models

This script combines Unsloth for fast training and Inspect AI for comprehensive evaluation,
providing an end-to-end workflow for the Google Gemma 3N Hackathon.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Import our modules
sys.path.append(str(Path(__file__).parent))
from src.models.gemma_model_unsloth import GemmaModelUnsloth
from src.evaluation.inspect_evaluator import GemmaInspectEvaluator, run_comprehensive_evaluation
from src.data.data_loader import DataLoader as GemmaDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedPipeline:
    """
    Unified pipeline for training with Unsloth and evaluating with Inspect AI.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.evaluator = None
        self.results = {
            "training": {},
            "evaluation": {},
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "config": config
            }
        }
        
    def prepare_dataset(self, data_path: str) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            data_path: Path to training data
            
        Returns:
            Prepared dataset
        """
        logger.info(f"Loading dataset from {data_path}")
        
        # Detect file format and load
        path = Path(data_path)
        if path.suffix == '.jsonl':
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'data' in data:
                    data = data['data']
        else:
            # Try loading as HuggingFace dataset
            dataset = load_dataset(data_path)
            return dataset['train']
            
        # Convert to Dataset
        dataset = Dataset.from_list(data)
        logger.info(f"Loaded {len(dataset)} samples")
        
        return dataset
    
    def train_with_unsloth(self, dataset: Dataset) -> str:
        """
        Train model using Unsloth optimizations.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Path to trained model
        """
        logger.info("🚀 Starting Unsloth training...")
        
        # Initialize Unsloth model
        self.model = GemmaModelUnsloth(
            model_name=self.config['model']['name'],
            max_seq_length=self.config['model']['max_seq_length'],
            dtype=None,  # Auto-detect
            load_in_4bit=self.config['model'].get('load_in_4bit', True)
        )
        
        # Load base model
        self.model.load_model()
        
        # Prepare for training with LoRA
        self.model.prepare_for_training(
            r=self.config['lora'].get('r', 16),
            lora_alpha=self.config['lora'].get('alpha', 16),
            lora_dropout=self.config['lora'].get('dropout', 0.05),
        )
        
        # Define formatting function for the dataset
        def formatting_func(example):
            if 'input' in example and 'output' in example:
                return f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
            elif 'text' in example:
                return example['text']
            else:
                return str(example)
        
        # Train model
        output_dir = self.config['training']['output_dir']
        self.model.train(
            dataset=dataset,
            output_dir=output_dir,
            num_epochs=self.config['training']['num_epochs'],
            per_device_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            warmup_steps=self.config['training']['warmup_steps'],
            learning_rate=self.config['training']['learning_rate'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            formatting_func=formatting_func,
        )
        
        # Save the final model
        final_model_path = f"{output_dir}/final_model"
        self.model.save_model(final_model_path, save_method="lora")
        
        logger.info(f"✅ Training completed! Model saved to {final_model_path}")
        
        self.results['training'] = {
            "status": "completed",
            "model_path": final_model_path,
            "dataset_size": len(dataset),
            "config": self.config['training']
        }
        
        return final_model_path
    
    def evaluate_with_inspect(self, model_path: str) -> Dict[str, Any]:
        """
        Evaluate model using Inspect AI.
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Evaluation results
        """
        logger.info("🔍 Starting Inspect AI evaluation...")
        
        # Initialize evaluator
        self.evaluator = GemmaInspectEvaluator(model_path)
        
        # Run comprehensive evaluation
        eval_results = run_comprehensive_evaluation(
            model_path=model_path,
            output_dir=self.config['evaluation']['output_dir']
        )
        
        # Add custom evaluations if specified
        if 'custom_eval_data' in self.config['evaluation']:
            logger.info("Running custom evaluations...")
            custom_results = self._run_custom_evaluations(
                self.config['evaluation']['custom_eval_data']
            )
            eval_results['custom'] = custom_results
        
        logger.info("✅ Evaluation completed!")
        
        self.results['evaluation'] = eval_results
        return eval_results
    
    def _run_custom_evaluations(self, custom_data_path: str) -> Dict[str, Any]:
        """
        Run custom evaluations.
        
        Args:
            custom_data_path: Path to custom evaluation data
            
        Returns:
            Custom evaluation results
        """
        # Load custom evaluation data
        with open(custom_data_path, 'r') as f:
            custom_data = json.load(f)
        
        results = {}
        
        for eval_name, eval_config in custom_data.items():
            task = self.evaluator.create_custom_eval(
                name=eval_name,
                dataset=eval_config['dataset'],
                system_prompt=eval_config.get('system_prompt', ''),
                scoring_method=eval_config.get('scoring_method', 'includes')
            )
            
            results[eval_name] = self.evaluator.run_evaluation(
                task,
                log_dir=f"{self.config['evaluation']['output_dir']}/custom/{eval_name}"
            )
        
        return results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of training and evaluation.
        
        Returns:
            Path to report file
        """
        logger.info("📊 Generating comprehensive report...")
        
        report_dir = Path(self.config['output_dir']) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate JSON report
        json_report_path = report_dir / "pipeline_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate Markdown report
        md_report = self._generate_markdown_report()
        md_report_path = report_dir / "pipeline_report.md"
        with open(md_report_path, 'w') as f:
            f.write(md_report)
        
        # Generate visualizations
        self._generate_visualizations(report_dir)
        
        logger.info(f"✅ Reports generated in {report_dir}")
        return str(report_dir)
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown report."""
        report = f"""# Gemma Model Training & Evaluation Report

Generated: {self.results['metadata']['timestamp']}

## Configuration

### Model
- **Base Model**: {self.config['model']['name']}
- **Sequence Length**: {self.config['model']['max_seq_length']}
- **4-bit Quantization**: {self.config['model'].get('load_in_4bit', True)}

### Training (Unsloth)
- **Epochs**: {self.config['training']['num_epochs']}
- **Batch Size**: {self.config['training']['batch_size']}
- **Learning Rate**: {self.config['training']['learning_rate']}
- **LoRA Rank**: {self.config['lora']['r']}

## Training Results

- **Status**: {self.results['training']['status']}
- **Dataset Size**: {self.results['training']['dataset_size']} samples
- **Model Path**: `{self.results['training']['model_path']}`

## Evaluation Results (Inspect AI)

### Overall Score: {self.results['evaluation'].get('overall_score', 0):.2%}

### Detailed Results:
"""
        
        for eval_name, eval_results in self.results['evaluation'].get('evaluations', {}).items():
            scores = eval_results['scores']
            report += f"""
#### {eval_name.capitalize()}
- **Accuracy**: {scores['accuracy']:.2%}
- **Mean Score**: {scores['mean_score']:.3f}
- **Std Dev**: {scores['std_score']:.3f}
- **Duration**: {eval_results['duration']:.2f}s
"""
        
        # Add custom evaluation results if present
        if 'custom' in self.results['evaluation']:
            report += "\n### Custom Evaluations:\n"
            for eval_name, eval_results in self.results['evaluation']['custom'].items():
                report += f"\n#### {eval_name}\n"
                report += f"- **Accuracy**: {eval_results['scores']['accuracy']:.2%}\n"
        
        return report
    
    def _generate_visualizations(self, report_dir: Path):
        """Generate visualization charts."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Extract evaluation scores
            eval_data = self.results['evaluation'].get('evaluations', {})
            if not eval_data:
                return
            
            names = list(eval_data.keys())
            scores = [data['scores']['accuracy'] for data in eval_data.values()]
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(names, scores)
            
            # Customize
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.title('Gemma Model Evaluation Results (Inspect AI)')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            
            # Add value labels
            for i, (name, score) in enumerate(zip(names, scores)):
                plt.text(i, score + 0.02, f'{score:.1%}', ha='center')
            
            plt.tight_layout()
            plt.savefig(report_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("📈 Visualizations saved")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualizations")
    
    def run_pipeline(self, data_path: str) -> Dict[str, Any]:
        """
        Run the complete training and evaluation pipeline.
        
        Args:
            data_path: Path to training data
            
        Returns:
            Complete results
        """
        logger.info("🎯 Starting Unified Pipeline")
        logger.info("=" * 50)
        
        try:
            # Step 1: Prepare dataset
            dataset = self.prepare_dataset(data_path)
            
            # Step 2: Train with Unsloth
            model_path = self.train_with_unsloth(dataset)
            
            # Step 3: Evaluate with Inspect AI
            eval_results = self.evaluate_with_inspect(model_path)
            
            # Step 4: Generate reports
            report_path = self.generate_report()
            
            logger.info("=" * 50)
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"📁 Results saved to: {report_path}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.results['error'] = str(e)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Unified Training & Evaluation Pipeline for Gemma Models"
    )
    
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to training data (JSON, JSONL, or HuggingFace dataset)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="Path to pipeline configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-2b-4bit",
        help="Model name (gemma-2b-4bit, gemma-7b-4bit, etc.)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./pipeline_outputs",
        help="Output directory for all results"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--eval-only",
        type=str,
        help="Skip training, only evaluate this model"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        "model": {
            "name": args.model,
            "max_seq_length": 2048,
            "load_in_4bit": True,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_r,
            "dropout": 0.05,
        },
        "training": {
            "output_dir": f"{args.output_dir}/models",
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "learning_rate": args.learning_rate,
            "logging_steps": 10,
            "save_steps": 500,
        },
        "evaluation": {
            "output_dir": f"{args.output_dir}/evaluations",
        },
        "output_dir": args.output_dir,
    }
    
    # Initialize pipeline
    pipeline = UnifiedPipeline(config)
    
    # Run pipeline
    if args.eval_only:
        # Evaluation only mode
        logger.info(f"Running evaluation only on {args.eval_only}")
        eval_results = pipeline.evaluate_with_inspect(args.eval_only)
        pipeline.generate_report()
    else:
        # Full pipeline
        results = pipeline.run_pipeline(args.data_path)
    
    logger.info("✨ All done!")


if __name__ == "__main__":
    main()