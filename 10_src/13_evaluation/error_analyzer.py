from typing import Dict, List
import numpy as np


class ErrorAnalyzer:

    def __init__(self):
        self.bleu = None
        self.chrf = None

        try:
            import evaluate
            self.bleu = evaluate.load("bleu")
            self.chrf = evaluate.load("chrf")
        except ImportError:
            pass

    def analyze_samples(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        error_samples = []

        for src, pred, ref in zip(sources, predictions, references):
            bleu_score = self._compute_sample_bleu(pred, ref)
            chrf_score = self._compute_sample_chrf(pred, ref)

            if bleu_score < 0.5:
                error_samples.append({
                    'source': src,
                    'prediction': pred,
                    'reference': ref,
                    'bleu': bleu_score,
                    'chrf': chrf_score,
                    'error_type': self._classify_error(src, pred, ref)
                })

        error_samples.sort(key=lambda x: x['bleu'])
        return error_samples[:100]

    def _compute_sample_bleu(self, prediction: str, reference: str) -> float:
        if not self.bleu:
            return 0.0

        try:
            result = self.bleu.compute(
                predictions=[prediction],
                references=[[reference]]
            )
            return result.get('bleu', 0.0)
        except:
            return 0.0

    def _compute_sample_chrf(self, prediction: str, reference: str) -> float:
        if not self.chrf:
            return 0.0

        try:
            result = self.chrf.compute(
                predictions=[prediction],
                references=[reference]
            )
            return result.get('score', 0.0)
        except:
            return 0.0

    def _classify_error(self, source: str, prediction: str, reference: str) -> str:
        pred_len = len(prediction.split())
        ref_len = len(reference.split())
        src_len = len(source.split())

        if pred_len < ref_len * 0.5:
            return "under_generation"
        elif pred_len > ref_len * 1.5:
            return "over_generation"
        elif len(set(prediction.split()) & set(reference.split())) < len(set(reference.split())) * 0.5:
            return "semantic_drift"
        else:
            return "minor_error"

    def compute_score_distribution(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        scores = []

        for pred, ref in zip(predictions, references):
            bleu_score = self._compute_sample_bleu(pred, ref)
            scores.append(bleu_score)

        if not scores:
            return {}

        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'quartiles': {
                'q25': float(np.percentile(scores, 25)),
                'q50': float(np.percentile(scores, 50)),
                'q75': float(np.percentile(scores, 75))
            }
        }