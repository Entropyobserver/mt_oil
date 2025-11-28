import os
import random
import numpy as np
import torch


class SeedManager:
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    @staticmethod
    def get_generator(seed: int = 42) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator