from omegaconf import DictConfig
from .base_trainer import BaseTrainer
from .lora_trainer import LoRATrainer
from .terminology_trainer import TerminologyLoRATrainer


class ModelFactory:

    _trainer_registry = {
        'base': BaseTrainer,
        'lora': LoRATrainer,
        'terminology_lora': TerminologyLoRATrainer
    }

    @classmethod
    def register_trainer(cls, name: str, trainer_class):
        cls._trainer_registry[name] = trainer_class

    @classmethod
    def create_trainer(
        cls,
        config: DictConfig,
        trainer_type: str = 'lora'
    ):
        if trainer_type not in cls._trainer_registry:
            raise ValueError(f"Unknown trainer type: {trainer_type}")

        trainer_class = cls._trainer_registry[trainer_type]

        trainer = trainer_class(
            model_name=config.model.pretrained_name,
            src_lang=config.model.tokenizer.src_lang,
            tgt_lang=config.model.tokenizer.tgt_lang
        )

        return trainer

    @classmethod
    def list_trainers(cls):
        return list(cls._trainer_registry.keys())
