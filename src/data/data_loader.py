"""
Data Loading and Preprocessing Module for Gemma Models

This module provides flexible data loading capabilities supporting multiple formats
including JSON, CSV, JSONL, and HuggingFace datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    """
    Custom PyTorch Dataset for text data.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            labels: Optional list of labels/targets
        """
        self.texts = texts
        self.labels = labels or texts  # For language modeling, labels = inputs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # For language modeling, we typically want labels to be the same as inputs
        # shifted by one position
        if text == label:  # Language modeling task
            labels = encoding["input_ids"].clone()
            # Replace padding token id with -100 for loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100
            encoding["labels"] = labels
        else:  # Sequence-to-sequence task
            label_encoding = self.tokenizer(
                label,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            labels = label_encoding["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = -100
            encoding["labels"] = labels
            
        return {key: val.squeeze(0) for key, val in encoding.items()}


class DataLoader:
    """
    Flexible data loader supporting multiple formats.
    """
    
    SUPPORTED_FORMATS = {".json", ".jsonl", ".csv", ".txt"}
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Initialize the data loader.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            # Assume it has a "data" or "examples" key
            for key in ["data", "examples", "items"]:
                if key in data:
                    data = data[key]
                    break
            else:
                # Convert dict to list of dicts
                data = [{"id": k, "text": v} for k, v in data.items()]
                
        return data
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        return df.to_dict("records")
    
    def load_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [{"text": line.strip()} for line in lines if line.strip()]
    
    def load_from_file(
        self,
        file_path: str,
        input_column: str = "text",
        target_column: Optional[str] = None,
        validation_split: float = 0.1,
    ) -> Union[TextDataset, tuple[TextDataset, TextDataset]]:
        """
        Load dataset from file.
        
        Args:
            file_path: Path to data file
            input_column: Column name for input text
            target_column: Column name for target text (optional)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dataset or tuple of (train_dataset, val_dataset)
        """
        file_path = Path(file_path)
        
        # Load data based on file format
        if file_path.suffix == ".json":
            data = self.load_json(file_path)
        elif file_path.suffix == ".jsonl":
            data = self.load_jsonl(file_path)
        elif file_path.suffix == ".csv":
            data = self.load_csv(file_path)
        elif file_path.suffix == ".txt":
            data = self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Extract texts and labels
        texts = []
        labels = []
        
        for item in data:
            if input_column in item:
                texts.append(item[input_column])
            elif "text" in item:
                texts.append(item["text"])
            elif "input" in item:
                texts.append(item["input"])
            else:
                # Try to use the first string value
                for value in item.values():
                    if isinstance(value, str):
                        texts.append(value)
                        break
                        
            if target_column and target_column in item:
                labels.append(item[target_column])
            elif target_column is None and "target" in item:
                labels.append(item["target"])
            elif target_column is None and "output" in item:
                labels.append(item["output"])
                
        # If no labels found, use texts as labels (language modeling)
        if not labels:
            labels = texts
            
        # Create train/val split if requested
        if validation_split > 0:
            split_idx = int(len(texts) * (1 - validation_split))
            
            train_dataset = TextDataset(
                texts[:split_idx],
                self.tokenizer,
                self.max_length,
                labels[:split_idx] if labels else None,
            )
            
            val_dataset = TextDataset(
                texts[split_idx:],
                self.tokenizer,
                self.max_length,
                labels[split_idx:] if labels else None,
            )
            
            return train_dataset, val_dataset
        else:
            return TextDataset(texts, self.tokenizer, self.max_length, labels)
            
    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        input_column: str = "text",
        target_column: Optional[str] = None,
        streaming: bool = False,
    ) -> HFDataset:
        """
        Load dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to load
            input_column: Column name for input text
            target_column: Column name for target text (optional)
            streaming: Whether to use streaming mode
            
        Returns:
            HuggingFace dataset
        """
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            inputs = examples[input_column]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            
            # Handle labels if present
            if target_column and target_column in examples:
                targets = examples[target_column]
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        targets,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                    )
                model_inputs["labels"] = labels["input_ids"]
            else:
                # For language modeling, labels are the same as inputs
                model_inputs["labels"] = model_inputs["input_ids"].copy()
                
            return model_inputs
            
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def create_dataloader(
        self,
        dataset: Union[TextDataset, HFDataset],
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from a dataset.
        
        Args:
            dataset: Dataset to create loader from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )
        
    def prepare_instruction_tuning_data(
        self,
        instructions: List[str],
        responses: List[str],
        system_prompt: Optional[str] = None,
    ) -> TextDataset:
        """
        Prepare data for instruction tuning.
        
        Args:
            instructions: List of instructions/prompts
            responses: List of corresponding responses
            system_prompt: Optional system prompt to prepend
            
        Returns:
            TextDataset formatted for instruction tuning
        """
        formatted_texts = []
        
        for instruction, response in zip(instructions, responses):
            # Format as conversation
            if system_prompt:
                text = f"{system_prompt}\n\nUser: {instruction}\n\nAssistant: {response}"
            else:
                text = f"User: {instruction}\n\nAssistant: {response}"
                
            formatted_texts.append(text)
            
        return TextDataset(
            formatted_texts,
            self.tokenizer,
            self.max_length,
        )