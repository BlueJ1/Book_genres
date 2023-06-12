import torch.nn as nn


# This class represents a feedforward neural network and defines a multilayer perceptron
# model with a variable number of hidden layers
class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """
    def __init__(self, model_size, dropout):
        super().__init__()
        layers = []
        for i in range(len(model_size) - 2):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(model_size[i], model_size[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(model_size[-2], model_size[-1]))
        layers.append(nn.Softmax(dim=1))  # Model gives probability of each class
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)
