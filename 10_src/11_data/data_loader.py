from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from .dataset import TranslationDataset, TranslationSample


class DataManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.data_dir = Path(config.paths.data_dir)
        self.splits_dir = self.data_dir / config.data.paths.splits
    
    def load_splits(
        self,
        tokenizer=None,
        use_subset: Optional[int] = None
    ) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
        
        train_path = self.data_dir / self.config.data.paths.train
        val_path = self.data_dir / self.config.data.paths.val
        test_path = self.data_dir / self.config.data.paths.test
        
        train_ds = TranslationDataset.from_json(
            train_path,
            tokenizer=tokenizer,
            max_length=self.config.model.tokenizer.max_length,
            src_lang=self.config.model.tokenizer.src_lang,
            tgt_lang=self.config.model.tokenizer.tgt_lang
        )
        
        val_ds = TranslationDataset.from_json(
            val_path,
            tokenizer=tokenizer,
            max_length=self.config.model.tokenizer.max_length,
            src_lang=self.config.model.tokenizer.src_lang,
            tgt_lang=self.config.model.tokenizer.tgt_lang
        )
        
        test_ds = TranslationDataset.from_json(
            test_path,
            tokenizer=tokenizer,
            max_length=self.config.model.tokenizer.max_length,
            src_lang=self.config.model.tokenizer.src_lang,
            tgt_lang=self.config.model.tokenizer.tgt_lang
        )
        
        if use_subset and use_subset < len(train_ds):
            train_ds = train_ds.subset(use_subset, seed=self.config.project.random_seed)
        
        return train_ds, val_ds, test_ds
    
    def create_splits(
        self,
        raw_data_path: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        save: bool = True
    ) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        full_ds = TranslationDataset.from_json(raw_data_path)
        samples = full_ds.samples
        
        train_samples, temp_samples = train_test_split(
            samples,
            test_size=(1 - train_ratio),
            random_state=self.config.project.random_seed
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(1 - val_size),
            random_state=self.config.project.random_seed
        )
        
        train_ds = TranslationDataset(train_samples)
        val_ds = TranslationDataset(val_samples)
        test_ds = TranslationDataset(test_samples)
        
        if save:
            self.splits_dir.mkdir(parents=True, exist_ok=True)
            train_ds.to_json(self.splits_dir / 'train.json')
            val_ds.to_json(self.splits_dir / 'val.json')
            test_ds.to_json(self.splits_dir / 'test.json')
        
        return train_ds, val_ds, test_ds
    
    def get_data_info(self) -> dict:
        try:
            train_ds, val_ds, test_ds = self.load_splits()
            
            return {
                'train': train_ds.get_statistics(),
                'val': val_ds.get_statistics(),
                'test': test_ds.get_statistics()
            }
        except FileNotFoundError:
            return {'error': 'Split files not found'}