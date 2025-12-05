import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset as TorchDataset


@dataclass
class TranslationSample:
    source: str
    target: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'metadata': self.metadata or {}
        }


class TranslationDataset(TorchDataset):
    def __init__(
        self,
        samples: List[TranslationSample],
        tokenizer=None,
        max_length: int = 128,
        src_lang: str = "nob_Latn",
        tgt_lang: str = "eng_Latn"
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        if self.tokenizer is None:
            return sample.to_dict()
        
        self.tokenizer.src_lang = self.src_lang
        inputs = self.tokenizer(
            sample.source,
            max_length=self.max_length,
            truncation=True,
            padding=False
        )
        
        self.tokenizer.tgt_lang = self.tgt_lang
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                sample.target,
                max_length=self.max_length,
                truncation=True,
                padding=False
            )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']
        }
    
    @classmethod
    def from_json(cls, path: Path, **kwargs) -> 'TranslationDataset':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = [
            TranslationSample(
                source=item.get('source', ''),
                target=item.get('target', ''),
                metadata=item.get('metadata')
            )
            for item in data
        ]
        
        return cls(samples, **kwargs)
    
    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([s.to_dict() for s in self.samples], f, ensure_ascii=False, indent=2)
    
    def filter_by_length(self, min_len: int = 3, max_len: int = 512) -> 'TranslationDataset':
        filtered = [
            s for s in self.samples
            if min_len <= len(s.source.split()) <= max_len
            and min_len <= len(s.target.split()) <= max_len
        ]
        return TranslationDataset(
            filtered,
            self.tokenizer,
            self.max_length,
            self.src_lang,
            self.tgt_lang
        )
    
    def subset(self, size: int, seed: int = 42) -> 'TranslationDataset':
        import random
        random.seed(seed)
        
        if size >= len(self.samples):
            return self
        
        indices = random.sample(range(len(self.samples)), size)
        subset_samples = [self.samples[i] for i in sorted(indices)]
        
        return TranslationDataset(
            subset_samples,
            self.tokenizer,
            self.max_length,
            self.src_lang,
            self.tgt_lang
        )
    
    def get_statistics(self) -> Dict:
        if not self.samples:
            return {
                'total': 0,
                'avg_source_length': 0,
                'avg_target_length': 0
            }
        
        src_lengths = [len(s.source.split()) for s in self.samples]
        tgt_lengths = [len(s.target.split()) for s in self.samples]
        
        return {
            'total': len(self.samples),
            'avg_source_length': sum(src_lengths) / len(src_lengths),
            'avg_target_length': sum(tgt_lengths) / len(tgt_lengths),
            'max_source_length': max(src_lengths),
            'max_target_length': max(tgt_lengths),
            'min_source_length': min(src_lengths),
            'min_target_length': min(tgt_lengths)
        }