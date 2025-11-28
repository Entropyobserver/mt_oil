import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sklearn.model_selection import train_test_split


class DataLoader:
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def load_json(self, file_path: Union[str, Path]) -> List[Dict]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
    
    def save_json(self, data: List[Dict], file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_splits(
        self,
        data: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        train_data, temp_data = train_test_split(
            data,
            test_size=(1 - train_ratio),
            random_state=self.random_seed
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=self.random_seed
        )
        
        return train_data, val_data, test_data
    
    def save_splits(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
        output_dir: Union[str, Path]
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_json(train_data, output_dir / "train.json")
        self.save_json(val_data, output_dir / "val.json")
        self.save_json(test_data, output_dir / "test.json")
    
    def load_splits(
        self,
        split_dir: Union[str, Path]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        split_dir = Path(split_dir)
        
        train_data = self.load_json(split_dir / "train.json")
        val_data = self.load_json(split_dir / "val.json")
        test_data = self.load_json(split_dir / "test.json")
        
        return train_data, val_data, test_data
    
    def sample_data(
        self,
        data: List[Dict],
        size: int,
        random: bool = True
    ) -> List[Dict]:
        if size >= len(data):
            return data
        
        if random:
            import random as rnd
            rnd.seed(self.random_seed)
            indices = rnd.sample(range(len(data)), size)
            return [data[i] for i in sorted(indices)]
        else:
            return data[:size]
    
    def get_statistics(self, data: List[Dict]) -> Dict:
        if not data:
            return {
                "size": 0,
                "avg_source_length": 0,
                "avg_target_length": 0
            }
        
        source_lengths = [
            len(item.get("source", "").split()) for item in data
        ]
        target_lengths = [
            len(item.get("target", "").split()) for item in data
        ]
        
        return {
            "size": len(data),
            "avg_source_length": sum(source_lengths) / len(source_lengths),
            "avg_target_length": sum(target_lengths) / len(target_lengths),
            "max_source_length": max(source_lengths),
            "max_target_length": max(target_lengths),
            "min_source_length": min(source_lengths),
            "min_target_length": min(target_lengths)
        }