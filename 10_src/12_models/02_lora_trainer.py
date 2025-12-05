import torch
import gc
from typing import Dict
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

from .base_trainer import BaseTrainer


class LoRATrainer(BaseTrainer):
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "nob_Latn",
        tgt_lang: str = "eng_Latn"
    ):
        super().__init__(model_name, src_lang, tgt_lang)
    
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
        
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        
        return model
    
    def train(self, train_data, val_data, config):
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        result = super().train(train_data, val_data, config)
        
        if config.get('save_final_model', True):
            final_path = Path(config['output_dir']) / 'final_model'
            final_path.mkdir(parents=True, exist_ok=True)
            result['trainer'].save_model(str(final_path))
            result['final_model_path'] = str(final_path)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result