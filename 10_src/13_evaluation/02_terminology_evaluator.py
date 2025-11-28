import json
import re
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
from rapidfuzz import fuzz
from nltk.stem.snowball import SnowballStemmer

from .base_evaluator import BaseEvaluator


class TerminologyEvaluator(BaseEvaluator):

    def __init__(
        self,
        glossary_path: Path,
        use_comet: bool = False,
        comet_model: str = "Unbabel/wmt22-comet-da"
    ):
        super().__init__(use_comet=use_comet, comet_model=comet_model)

        self.glossary = self._load_glossary(glossary_path)
        self.inverse_glossary = {v.lower(): k.lower() for k, v in self.glossary.items()}
        self.stemmer = SnowballStemmer("english")

    def _load_glossary(self, path: Path) -> Dict[str, str]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            glossary = {}
            for entry in data:
                source_term = entry.get('en', '').strip()
                target_term = entry.get('no', '').strip()
                if source_term and target_term:
                    glossary[source_term.lower()] = target_term.lower()

            return glossary
        except Exception as e:
            print(f"Error loading glossary: {e}")
            return {}

    def _extract_source_terms(self, source_text: str) -> List[str]:
        source_lower = source_text.lower()
        found_terms = []

        for source_term in self.glossary.keys():
            pattern = r'\b' + re.escape(source_term) + r'\b'
            if re.search(pattern, source_lower, re.IGNORECASE):
                found_terms.append(source_term)

        return found_terms

    def _find_target_term_in_prediction(
        self,
        target_term: str,
        prediction_text: str,
        threshold: float = 0.80
    ) -> bool:
        prediction_lower = prediction_text.lower()

        pattern = r'\b' + re.escape(target_term) + r'\b'
        if re.search(pattern, prediction_lower):
            return True

        similarity = fuzz.partial_ratio(target_term, prediction_lower) / 100.0
        if similarity >= threshold:
            return True

        return False

    def calculate_term_accuracy(
        self,
        source: str,
        prediction: str,
        reference: str
    ) -> Dict:
        source_terms = self._extract_source_terms(source)

        if not source_terms:
            return {
                'term_precision': 1.0,
                'term_recall': 1.0,
                'term_f1': 1.0,
                'term_count': 0
            }

        expected_target_terms = set()
        for src_term in source_terms:
            target_term = self.glossary.get(src_term, '')
            if target_term:
                expected_target_terms.add(target_term)

        predicted_target_terms = set()
        for expected_term in expected_target_terms:
            if self._find_target_term_in_prediction(expected_term, prediction):
                predicted_target_terms.add(expected_term)

        correct_terms = expected_target_terms.intersection(predicted_target_terms)
        correct_count = len(correct_terms)
        predicted_count = len(predicted_target_terms)
        expected_count = len(expected_target_terms)

        precision = correct_count / predicted_count if predicted_count > 0 else 1.0
        recall = correct_count / expected_count if expected_count > 0 else 1.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'term_precision': precision,
            'term_recall': recall,
            'term_f1': f1_score,
            'term_count': expected_count,
            'correct_terms': correct_count
        }

    def evaluate_corpus_terminology(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        all_results = []

        for src, pred, ref in zip(sources, predictions, references):
            result = self.calculate_term_accuracy(src, pred, ref)
            if result['term_count'] > 0:
                all_results.append(result)

        if not all_results:
            return {
                'term_accuracy': 0.0,
                'term_precision': 0.0,
                'term_recall': 0.0,
                'term_f1': 0.0,
                'sentences_with_terms': 0
            }

        total_correct = sum(r['correct_terms'] for r in all_results)
        total_terms = sum(r['term_count'] for r in all_results)

        avg_precision = np.mean([r['term_precision'] for r in all_results])
        avg_recall = np.mean([r['term_recall'] for r in all_results])
        avg_f1 = np.mean([r['term_f1'] for r in all_results])

        term_accuracy = total_correct / total_terms if total_terms > 0 else 0.0

        return {
            'term_accuracy': term_accuracy,
            'term_precision': avg_precision,
            'term_recall': avg_recall,
            'term_f1': avg_f1,
            'sentences_with_terms': len(all_results)
        }

    def evaluate_all(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        base_metrics = super().evaluate_all(sources, predictions, references)
        term_metrics = self.evaluate_corpus_terminology(sources, predictions, references)
        base_metrics.update(term_metrics)
        return base_metrics
