from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTrainer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._build_model()
        self.optimizer = self._set_optimizer()
        self.criterion = self._set_train_criterion()

    @abstractmethod
    def _build_model(self) -> Any:
        """Initialize the neural network architecture."""
        pass

    @abstractmethod
    def _set_optimizer(self) -> Any:
        """Define the optimizer (e.g., Adam, SGD)."""
        pass

    @abstractmethod
    def _set_train_criterion(self) -> Any:
        """Define the loss function."""
        pass

    @abstractmethod
    def train_epoch(self, **kwargs) -> Any:
        """Logic for a single training epoch. Returns average loss."""
        pass

    @abstractmethod
    def val(self, **kwargs) -> Any:
        """Logic for evaluation. Returns a dictionary of metrics."""
        pass
    
    @abstractmethod
    def train_loop(self, **kwargs) -> Any:
        """Define training loop"""
        