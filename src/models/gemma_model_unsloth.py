"""
Optimized Google Gemma Model Wrapper using Unsloth

This module provides a faster, more memory-efficient wrapper for Gemma models
using Unsloth optimizations for 2-5x faster training and 70% less memory usage.
"""

import os
from typing import Dict, List, Optional, Union
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: Unsloth not installed. Install with: pip install unsloth")


class GemmaModelUnsloth:
    """
    Optimized wrapper for Google Gemma models using Unsloth.
    Provides 2-5x faster training and 70% less memory usage.
    """
    
    SUPPORTED_MODELS = {
        "gemma-2b": "unsloth/gemma-2b",
        "gemma-7b": "unsloth/gemma-7b", 
        "gemma-2b-it": "unsloth/gemma-2b-it-bnb-4bit",
        "gemma-7b-it": "unsloth/gemma-7b-it-bnb-4bit",
        # 4-bit quantized versions for even better efficiency
        "gemma-2b-4bit": "unsloth/gemma-2b-bnb-4bit",
        "gemma-7b-4bit": "unsloth/gemma-7b-bnb-4bit",
    }
    
    def __init__(
        self,
        model_name: str = "gemma-2b-4bit",
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
    ):
        """
        Initialize the Unsloth-optimized Gemma model.
        
        Args:
            model_name: Model variant to use
            max_seq_length: Maximum sequence length
            dtype: Data type (None for auto-detection)
            load_in_4bit: Use 4-bit quantization
        """
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is not installed. Install it with:\n"
                "pip install unsloth"
            )
            
        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Load the model with Unsloth optimizations.
        
        Args:
            checkpoint_path: Path to fine-tuned checkpoint (optional)
        """
        model_path = checkpoint_path or self.model_id
        
        # Load with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            # Unsloth specific optimizations
            trust_remote_code=True,
            use_cache=False,  # Disable KV cache for training efficiency
        )
        
        print(f"Model {self.model_name} loaded with Unsloth optimizations!")
        
    def prepare_for_training(self, r: int = 16, lora_alpha: int = 16, 
                           lora_dropout: float = 0.05, target_modules: Optional[List[str]] = None):
        """
        Prepare model for training with LoRA (Low-Rank Adaptation).
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Modules to apply LoRA to (None for auto)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Apply LoRA with Unsloth optimizations
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            use_gradient_checkpointing=True,  # Memory efficiency
            random_state=42,
        )
        
        print("Model prepared for training with LoRA!")
        
    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./outputs",
        num_epochs: int = 3,
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_steps: int = 500,
        max_steps: int = -1,
        formatting_func: Optional[callable] = None,
    ):
        """
        Train the model using Unsloth-optimized SFTTrainer.
        
        Args:
            dataset: Training dataset
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            per_device_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            learning_rate: Learning rate
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            max_steps: Maximum training steps (-1 for full epochs)
            formatting_func: Function to format dataset examples
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Prepare for training if not already done
        if not hasattr(self.model, 'peft_config'):
            self.prepare_for_training()
            
        # Training arguments optimized for Unsloth
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            max_steps=max_steps,
            # Unsloth optimizations
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",  # 8-bit optimizer
            seed=42,
            report_to="none",  # Disable wandb/tensorboard for speed
            save_strategy="steps",
            save_total_limit=2,
        )
        
        # Default formatting function for instruction tuning
        if formatting_func is None:
            def formatting_func(example):
                if "input" in example and "output" in example:
                    return f"User: {example['input']}\n\nAssistant: {example['output']}"
                elif "text" in example:
                    return example["text"]
                else:
                    return str(example)
        
        # Initialize SFTTrainer with Unsloth optimizations
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            formatting_func=formatting_func,
            max_seq_length=self.max_seq_length,
            packing=True,  # Efficient packing of short sequences
        )
        
        # Train with automatic mixed precision
        trainer.train()
        
        print(f"Training completed! Model saved to {output_dir}")
        
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> List[str]:
        """
        Generate text using the model.
        
        Args:
            prompts: Input prompt(s)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            use_cache: Use KV cache for generation
            
        Returns:
            List of generated texts
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Enable cache for generation
        FastLanguageModel.for_inference(self.model)
        
        generated_texts = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            ).to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    use_cache=use_cache,
                )
                
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            generated_text = text[len(prompt):].strip()
            generated_texts.append(generated_text)
            
        return generated_texts
    
    def save_model(self, save_path: str, save_method: str = "lora"):
        """
        Save the model efficiently.
        
        Args:
            save_path: Directory to save the model
            save_method: "lora" (only LoRA weights) or "merged_16bit" (full model)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Nothing to save.")
            
        if save_method == "lora":
            # Save only LoRA adapters (small size)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        elif save_method == "merged_16bit":
            # Save merged model in 16-bit (larger but standalone)
            self.model.save_pretrained_merged(save_path, self.tokenizer, save_method="merged_16bit")
        else:
            raise ValueError(f"Unknown save method: {save_method}")
            
        print(f"Model saved to {save_path} using {save_method} method")
        
    def push_to_hub(self, repo_name: str, save_method: str = "lora"):
        """
        Push model to HuggingFace Hub.
        
        Args:
            repo_name: Repository name on HuggingFace
            save_method: "lora" or "merged_16bit"
        """
        if save_method == "lora":
            self.model.push_to_hub(repo_name, token=True)
            self.tokenizer.push_to_hub(repo_name, token=True)
        else:
            self.model.push_to_hub_merged(repo_name, self.tokenizer, save_method="merged_16bit", token=True)
            
        print(f"Model pushed to hub: {repo_name}")