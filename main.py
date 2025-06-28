import argparse
import logging
from pathlib import Path

from src.models.gemma_model import GemmaModel
from src.data.data_loader import DataLoader
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Google Gemma 3N Hackathon Project')
    parser.add_argument('--task', type=str, required=True, 
                        choices=['train', 'evaluate', 'predict'],
                        help='Task to perform')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_path', type=str, 
                        help='Path to saved model checkpoint')
    parser.add_argument('--data_path', type=str,
                        help='Path to input data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory for outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize model
    model = GemmaModel(config)
    
    if args.task == 'train':
        logger.info('Starting training...')
        # TODO: Implement training logic
        pass
    
    elif args.task == 'evaluate':
        logger.info('Starting evaluation...')
        # TODO: Implement evaluation logic
        pass
    
    elif args.task == 'predict':
        logger.info('Starting prediction...')
        # TODO: Implement prediction logic
        pass


if __name__ == '__main__':
    main()