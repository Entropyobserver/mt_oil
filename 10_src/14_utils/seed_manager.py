import random
import numpy as np
import torch
import os


class SeedManager:
    
    @staticmethod
    def set_seed(seed: int, deterministic: bool = True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    @staticmethod
    def get_generator(seed: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed)
        return g