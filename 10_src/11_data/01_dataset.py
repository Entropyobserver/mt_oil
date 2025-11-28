import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


@dataclass
class TranslationSample:
    source: str
    target: str
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            **self.metadata
        }


class TranslationDataset(Dataset):
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 128,
        src_lang: str = "nob_Latn",
        tgt_lang: str = "eng_Latn"
    ):
        self.data = [TranslationSample(**item) for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        self.tokenizer.src_lang = self.src_lang
        source_encoded = self.tokenizer(
            sample.source,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        self.tokenizer.tgt_lang = self.tgt_lang
        target_encoded = self.tokenizer(
            sample.target,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": source_encoded["input_ids"].squeeze(),
            "attention_mask": source_encoded["attention_mask"].squeeze(),
            "labels": target_encoded["input_ids"].squeeze()
        }
    
    @classmethod
    def from_json(
        cls,
        file_path: Union[str, Path],
        tokenizer,
        max_length: int = 128,
        src_lang: str = "nob_Latn",
        tgt_lang: str = "eng_Latn"
    ) -> "TranslationDataset":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, tokenizer, max_length, src_lang, tgt_lang)
    
    def to_huggingface(self) -> HFDataset:
        data_dict = [sample.to_dict() for sample in self.data]
        return HFDataset.from_list(data_dict)
    
    def get_statistics(self) -> Dict:
        source_lengths = [len(s.source.split()) for s in self.data]
        target_lengths = [len(s.target.split()) for s in self.data]
        
        return {
            "num_samples": len(self.data),
            "avg_source_length": sum(source_lengths) / len(source_lengths),
            "avg_target_length": sum(target_lengths) / len(target_lengths),
            "max_source_length": max(source_lengths),
            "max_target_length": max(target_lengths),
            "min_source_length": min(source_lengths),
            "min_target_length": min(target_lengths)
        }