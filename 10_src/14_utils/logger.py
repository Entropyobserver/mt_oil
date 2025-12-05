import logging
import sys
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig


class Logger:
    
    _instances = {}
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        config: Optional[DictConfig] = None,
        log_file: Optional[Path] = None
    ) -> logging.Logger:
        
        if name in cls._instances:
            return cls._instances[name]
        
        logger = logging.getLogger(name)
        
        if config:
            level = getattr(logging, config.logging.level)
            logger.setLevel(level)
            
            formatter = logging.Formatter(config.logging.format)
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        else:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        
        logger.propagate = False
        
        cls._instances[name] = logger
        return logger
    
    @classmethod
    def reset(cls):
        for logger in cls._instances.values():
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        cls._instances = {}