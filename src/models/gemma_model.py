"""
Google Gemma Model Wrapper for the Hackathon Project

This module provides a wrapper around Google's Gemma models from HuggingFace,
supporting both 2B and 7B parameter versions with training and inference capabilities.
"""

import os
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator


class GemmaModel:
    """
    Wrapper class for Google Gemma models providing easy-to-use interface
    for loading, training, and inference.
    """
    
    SUPPORTED_MODELS = {
        "gemma-2b": "google/gemma-2b",
        "gemma-7b": "google/gemma-7b",
        "gemma-2b-it": "google/gemma-2b-it",  # Instruction-tuned variant
        "gemma-7b-it": "google/gemma-7b-it",  # Instruction-tuned variant
    }
    
    def __init__(
        self,
        model_name: str = "gemma-2b",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        use_flash_attention: bool = True,
    ):
        """
        Initialize the Gemma model wrapper.
        
        Args:
            model_name: Name of the model variant (gemma-2b, gemma-7b, etc.)
            device: Device to load model on (cuda/cpu/auto)
            load_in_8bit: Whether to load model in 8-bit precision
            use_flash_attention: Whether to use Flash Attention 2
        """
        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.use_flash_attention = use_flash_attention
        
        self.model = None
        self.tokenizer = None
        self.accelerator = Accelerator()
        
    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Load the model and tokenizer from HuggingFace or local checkpoint.
        
        Args:
            checkpoint_path: Path to local checkpoint (optional)
        """
        model_path = checkpoint_path or self.model_id
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model loading arguments
        model_kwargs = {
            "device_map": "auto" if self.device == "auto" else None,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": True,
        }
        
        # Add 8-bit loading if requested
        if self.load_in_8bit and torch.cuda.is_available():
            model_kwargs["load_in_8bit"] = True
            
        # Add Flash Attention if available
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if self.device != "auto" and not self.load_in_8bit:
            self.model = self.model.to(self.device)
            
        print(f"Model {self.model_name} loaded successfully!")
        
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt(s)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated texts
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Tokenize inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        # Move to device
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )
            
        # Decode outputs
        generated_texts = []
        for i in range(0, len(outputs), num_return_sequences):
            batch_outputs = outputs[i:i + num_return_sequences]
            for output in batch_outputs:
                # Skip the input tokens
                generated_ids = output[inputs["input_ids"].shape[-1]:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_texts.append(text.strip())
                
        return generated_texts
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./checkpoints",
        num_epochs: int = 3,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        learning_rate: float = 2e-5,
        fp16: bool = True,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        save_total_limit: int = 2,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs
    ):
        """
        Fine-tune the model on a dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            per_device_batch_size: Batch size per device
            gradient_accumulation_steps: Number of gradient accumulation steps
            warmup_steps: Number of warmup steps
            learning_rate: Learning rate
            fp16: Whether to use mixed precision training
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            save_total_limit: Maximum number of checkpoints to keep
            resume_from_checkpoint: Path to resume training from
            **kwargs: Additional training arguments
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=fp16 and torch.cuda.is_available(),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to=["tensorboard"],
            push_to_hub=False,
            **kwargs
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Gemma is a causal LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Training completed! Model saved to {output_dir}")
        
    def save_model(self, save_path: str):
        """
        Save the model and tokenizer to disk.
        
        Args:
            save_path: Directory to save the model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Nothing to save.")
            
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        
    def push_to_hub(self, repo_name: str, private: bool = True):
        """
        Push the model to HuggingFace Hub.
        
        Args:
            repo_name: Name of the repository
            private: Whether to make the repo private
        """
        if self.model is None:
            raise ValueError("Model not loaded. Nothing to push.")
            
        self.model.push_to_hub(repo_name, private=private)
        self.tokenizer.push_to_hub(repo_name, private=private)
        print(f"Model pushed to hub: {repo_name}")