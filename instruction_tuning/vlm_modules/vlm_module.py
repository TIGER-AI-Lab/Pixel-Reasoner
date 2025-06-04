from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import torch


class VLMBaseModule(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_vlm_key(self):
        pass

    @abstractmethod
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        pass
    
    def get_model(self):
        pass

    def get_processor(self):
        pass

    def post_model_init(self, model, processing_class):
        pass


    @abstractmethod
    def get_processing_class(self):
        pass

    @abstractmethod
    def get_vision_modules_keywords(self):
        pass









