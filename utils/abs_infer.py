from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseInfer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._build_model()
        self.criterion = self._set_evaluation_criterion()


    @abstractmethod
    def _build_model(self) -> Any:
        """Initialize the neural network architecture. Load pretrained weights if necessary."""
        pass

    @abstractmethod
    def _set_evaluation_criterion(self) -> Any:
        """Define the evaluation function."""
        pass

    @abstractmethod
    def infer(self, input, **kwargs) -> Any:
        """Logic for inference. Returns predictions."""
        pass

    @abstractmethod
    def evaluate(self, predictions, gt, **kwargs) -> Any:
        """
        Logic for evaluation. Takes predictions and ground truth as input.
        Returns a dictionary of metrics. 
        Expected to be used after infer.
        """
        pass