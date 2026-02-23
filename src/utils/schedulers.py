from abc import ABCMeta, abstractmethod
import numpy as np


class BaseScheduler(metaclass=ABCMeta):
    name = "base_scheduler"

    def __init__(self, initial_value, final_value, max_step):
        super().__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.max_step = max_step
    
    @classmethod
    def get_name(cls):
        return cls.name
    
    @abstractmethod
    def get_value(self, step) -> float:
        pass
    

class LinearScheduler(BaseScheduler):
    name = "linear_scheduler"

    def __init__(self, initial_value, final_value, max_step, **kwargs):
        super().__init__(initial_value, final_value, max_step)
        self.interval = (final_value - initial_value) / max_step

    def get_value(self, step):
        step = min(step, self.max_step)
        return self.initial_value + self.interval * step
    
class CosineScheduler(BaseScheduler):
    name = "cosine_scheduler"

    def __init__(self, initial_value, final_value, max_step, **kwargs):
        super().__init__(initial_value, final_value, max_step)

    def get_value(self, step):
        step = min(step, self.max_step)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.max_step))
        return self.final_value + (self.initial_value - self.final_value) * cosine_decay
    
class ExponentialScheduler(BaseScheduler):
    name = "exponential_scheduler"

    def __init__(self, initial_value, final_value, max_step, reverse=False, **kwargs):
        super().__init__(initial_value, final_value, max_step)
        self.reverse = reverse
        if self.reverse:
            self.initial_value = 1 - initial_value
            self.final_value = 1 - final_value
        self.interval = (np.log(self.final_value) - np.log(self.initial_value)) / max_step

    def get_value(self, step):
        step = min(step, self.max_step)
        start = np.log(self.initial_value)
        
        if self.reverse:
            return 1 - np.exp(start + self.interval * step)
        else:
            return np.exp(start + self.interval * step)