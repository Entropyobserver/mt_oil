import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate


class BaseTrainer(ABC):
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "nob_Latn",
        tgt_lang: str = "eng_Latn"
    ):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    @abstractmethod
    def setup_model(self, **kwargs):
        pass
    
    def tokenize_function(self, examples: Dict) -> Dict:
        self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            examples["source"],
            max_length=128,
            truncation=True,
            padding=False
        )
        
        self.tokenizer.tgt_lang = self.tgt_lang
        labels = self.tokenizer(
            examples["target"],
            max_length=128,
            truncation=True,
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics(self, eval_pred) -> Dict:
        predictions, labels = eval_pred
        
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
        
        decoded_preds = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )
        
        bleu_result = self.bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        chrf_result = self.chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        
        return {
            "bleu": bleu_result["bleu"],
            "chrf": chrf_result["score"]
        }
    
    def prepare_datasets(
        self,
        train_data: List[Dict],
        val_data: List[Dict]
    ) -> tuple[Dataset, Dataset]:
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return train_dataset, val_dataset
    
    def create_training_args(self, config: Dict) -> Seq2SeqTrainingArguments:
        
        eval_steps = config.get("eval_steps", 200)
        save_steps = config.get("save_steps", 400)
        
        if save_steps % eval_steps != 0:
            save_steps = eval_steps * 2
        
        return Seq2SeqTrainingArguments(
            output_dir=config.get("output_dir", "output"),
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 4),
            per_device_eval_batch_size=config.get("batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_steps=config.get("warmup_steps", 100),
            learning_rate=config.get("learning_rate", 5e-4),
            weight_decay=config.get("weight_decay", 0.01),
            logging_steps=config.get("logging_steps", 50),
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=config.get("save_total_limit", 2),
            load_best_model_at_end=True,
            metric_for_best_model=config.get("metric_for_best_model", "bleu"),
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=config.get("generation_max_length", 128),
            fp16=config.get("fp16", True),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=config.get("report_to", []),
            dataloader_num_workers=config.get("dataloader_num_workers", 0),
            max_grad_norm=config.get("max_grad_norm", 1.0)
        )
    
    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        config: Dict
    ) -> Dict:
        
        model = self.setup_model(**config)
        
        train_dataset, val_dataset = self.prepare_datasets(
            train_data,
            val_data
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model,
            padding=True
        )
        
        training_args = self.create_training_args(config)
        
        callbacks = []
        if config.get("early_stopping_patience"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config["early_stopping_patience"],
                    early_stopping_threshold=config.get(
                        "early_stopping_threshold",
                        0.001
                    )
                )
            )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        trainer.train()
        eval_result = trainer.evaluate()
        
        return {
            "model": model,
            "trainer": trainer,
            "bleu": eval_result.get("eval_bleu", 0.0),
            "chrf": eval_result.get("eval_chrf", 0.0),
            "loss": eval_result.get("eval_loss", 0.0)
        }