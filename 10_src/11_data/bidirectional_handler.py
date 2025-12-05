from typing import Tuple
from .dataset import TranslationDataset, TranslationSample


class BidirectionalHandler:
    def __init__(
        self,
        forward_src_lang: str = "nob_Latn",
        forward_tgt_lang: str = "eng_Latn"
    ):
        self.forward_src_lang = forward_src_lang
        self.forward_tgt_lang = forward_tgt_lang
    
    def reverse_dataset(self, dataset: TranslationDataset) -> TranslationDataset:
        reversed_samples = [
            TranslationSample(
                source=sample.target,
                target=sample.source,
                metadata=sample.metadata
            )
            for sample in dataset.samples
        ]
        
        return TranslationDataset(
            reversed_samples,
            tokenizer=dataset.tokenizer,
            max_length=dataset.max_length,
            src_lang=self.forward_tgt_lang,
            tgt_lang=self.forward_src_lang
        )
    
    def create_bidirectional_dataset(
        self,
        dataset: TranslationDataset
    ) -> TranslationDataset:
        
        reversed_dataset = self.reverse_dataset(dataset)
        combined_samples = dataset.samples + reversed_dataset.samples
        
        return TranslationDataset(
            combined_samples,
            tokenizer=dataset.tokenizer,
            max_length=dataset.max_length
        )
    
    def split_by_direction(
        self,
        dataset: TranslationDataset
    ) -> Tuple[TranslationDataset, TranslationDataset]:
        
        forward_samples = []
        reverse_samples = []
        
        for sample in dataset.samples:
            metadata = sample.metadata or {}
            if metadata.get('direction') == 'reverse':
                reverse_samples.append(sample)
            else:
                forward_samples.append(sample)
        
        forward_ds = TranslationDataset(
            forward_samples,
            dataset.tokenizer,
            dataset.max_length,
            self.forward_src_lang,
            self.forward_tgt_lang
        )
        
        reverse_ds = TranslationDataset(
            reverse_samples,
            dataset.tokenizer,
            dataset.max_length,
            self.forward_tgt_lang,
            self.forward_src_lang
        )
        
        return forward_ds, reverse_ds