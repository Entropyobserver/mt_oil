import torch
from typing import Dict, List
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSeq2SeqLM

from .lora_trainer import LoRATrainer


class TerminologyLoRATrainer(LoRATrainer):

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "nob_Latn",
        tgt_lang: str = "eng_Latn"
    ):
        super().__init__(model_name, src_lang, tgt_lang)
        self._prepare_tokenizer()

    def _prepare_tokenizer(self):
        special_tokens = {
            'additional_special_tokens': ['<TERM>', '</TERM>']
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens to tokenizer")

    def setup_model(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: list = None,
        **kwargs
    ):
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        for param in base_model.parameters():
            param.requires_grad = False

        vocab_size = len(self.tokenizer)
        model_vocab_size = base_model.config.vocab_size

        if vocab_size > model_vocab_size:
            base_model.resize_token_embeddings(vocab_size)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none"
        )

        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        return model
