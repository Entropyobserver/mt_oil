from typing import Dict, List
from enum import Enum


class TranslationDirection(Enum):
    NOB_TO_ENG = "nob_to_eng"
    ENG_TO_NOB = "eng_to_nob"


class BidirectionalDataHandler:
    
    def __init__(
        self,
        nob_lang_code: str = "nob_Latn",
        eng_lang_code: str = "eng_Latn"
    ):
        self.nob_lang_code = nob_lang_code
        self.eng_lang_code = eng_lang_code
    
    def get_direction_config(
        self,
        direction: TranslationDirection
    ) -> Dict[str, str]:
        if direction == TranslationDirection.NOB_TO_ENG:
            return {
                "src_lang": self.nob_lang_code,
                "tgt_lang": self.eng_lang_code,
                "source_field": "source",
                "target_field": "target"
            }
        elif direction == TranslationDirection.ENG_TO_NOB:
            return {
                "src_lang": self.eng_lang_code,
                "tgt_lang": self.nob_lang_code,
                "source_field": "target",
                "target_field": "source"
            }
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def reverse_data(self, data: List[Dict]) -> List[Dict]:
        reversed_data = []
        for item in data:
            reversed_item = {
                "source": item.get("target", ""),
                "target": item.get("source", "")
            }
            if "metadata" in item:
                reversed_item["metadata"] = item["metadata"]
            reversed_data.append(reversed_item)
        return reversed_data
    
    def create_bidirectional_dataset(
        self,
        data: List[Dict]
    ) -> List[Dict]:
        forward_data = data
        reverse_data = self.reverse_data(data)
        return forward_data + reverse_data
    
    def split_by_direction(
        self,
        bidirectional_data: List[Dict]
    ) -> tuple[List[Dict], List[Dict]]:
        mid = len(bidirectional_data) // 2
        forward_data = bidirectional_data[:mid]
        reverse_data = bidirectional_data[mid:]
        return forward_data, reverse_data