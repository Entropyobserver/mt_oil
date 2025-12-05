import json
import re
from pathlib import Path
from typing import Dict, List, Set
from .dataset import TranslationDataset, TranslationSample


class TerminologyHandler:
    def __init__(self, glossary_path: Path):
        self.glossary = self._load_glossary(glossary_path)
        self.term_start_token = '<TERM>'
        self.term_end_token = '</TERM>'
    
    def _load_glossary(self, path: Path) -> Dict[str, str]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        glossary = {}
        for entry in data:
            source_term = entry.get('en', '').strip()
            target_term = entry.get('no', '').strip()
            if source_term and target_term:
                glossary[source_term.lower()] = target_term.lower()
        
        return glossary
    
    def mark_terms_in_text(self, text: str, is_source: bool = True) -> str:
        terms_dict = self.glossary if is_source else {v: k for k, v in self.glossary.items()}
        
        sorted_terms = sorted(terms_dict.keys(), key=len, reverse=True)
        
        marked_text = text
        for term in sorted_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            replacement = f'{self.term_start_token}{term}{self.term_end_token}'
            marked_text = re.sub(pattern, replacement, marked_text, flags=re.IGNORECASE)
        
        return marked_text
    
    def create_terminology_dataset(
        self,
        dataset: TranslationDataset,
        mark_source: bool = True,
        mark_target: bool = True
    ) -> TranslationDataset:
        
        marked_samples = []
        for sample in dataset.samples:
            source = self.mark_terms_in_text(sample.source, is_source=True) if mark_source else sample.source
            target = self.mark_terms_in_text(sample.target, is_source=False) if mark_target else sample.target
            
            marked_samples.append(
                TranslationSample(
                    source=source,
                    target=target,
                    metadata={**(sample.metadata or {}), 'terminology_marked': True}
                )
            )
        
        return TranslationDataset(
            marked_samples,
            dataset.tokenizer,
            dataset.max_length,
            dataset.src_lang,
            dataset.tgt_lang
        )
    
    def extract_terms_from_text(self, text: str, is_source: bool = True) -> Set[str]:
        text_lower = text.lower()
        terms_dict = self.glossary if is_source else {v: k for k, v in self.glossary.items()}
        
        found_terms = set()
        for term in terms_dict.keys():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_terms.add(term)
        
        return found_terms
    
    def get_terminology_statistics(self, dataset: TranslationDataset) -> Dict:
        total_terms = 0
        samples_with_terms = 0
        
        for sample in dataset.samples:
            source_terms = self.extract_terms_from_text(sample.source, is_source=True)
            if source_terms:
                samples_with_terms += 1
                total_terms += len(source_terms)
        
        return {
            'total_samples': len(dataset.samples),
            'samples_with_terms': samples_with_terms,
            'total_term_occurrences': total_terms,
            'avg_terms_per_sample': total_terms / len(dataset.samples) if dataset.samples else 0,
            'glossary_size': len(self.glossary)
        }