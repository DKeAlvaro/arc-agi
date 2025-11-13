from abc import ABC, abstractmethod
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class ARCModel(ABC):
    @abstractmethod
    def train(self, train_examples):
        """Train the model on given training examples"""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Predict output for given input data"""
        pass


class DummyModel(ARCModel):
    def train(self, train_examples):
        self.train_examples = train_examples

    def predict(self, input_data):
        # Simple dummy prediction
        rows = len(input_data)
        cols = len(input_data[0]) if rows > 0 else 0
        return [[0 for _ in range(cols)] for _ in range(rows)]

