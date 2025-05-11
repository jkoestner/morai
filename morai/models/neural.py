"""
Creates neural models for forecasting mortality rates.

The neural network does not perform well on small tabular data. The relationships
it finds do not line up with the relationships in the data.

"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


class Neural(nn.Module):
    """
    Neural network model.

    The nn.Module class is needed to ensure proper layer registration and
    parameter tracking. This is inherited in the "super" line.

    Notes
    -----
    The model architecture is:
        fc1 -> relu1 -> fc2 -> relu2 -> fc3 -> relu3 -> output
    the forward function does use Softplus activation to ensure non-negative
    output

    """

    def __init__(self) -> None:
        """Initialize the model."""
        super(Neural, self).__init__()
        # first layer
        self.fc1 = None
        # first activation
        self.relu1 = None
        # second layer
        self.fc2 = None
        # second activation
        self.relu2 = None
        # third layer
        self.fc3 = None
        # third activation
        self.relu3 = None
        # output layer
        self.output = None

    def setup_model(self, X_train: pd.DataFrame) -> None:
        """
        Model architecture setup.

        Parameters
        ----------
        X_train : pd.DataFrame
            A DataFrame containing the data to structure.

        """
        input_size = X_train.shape[1]
        self.fc1 = nn.Linear(input_size, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(10, 1)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        weights_train: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        lr: float = 0.001,
    ) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data
        y_train : pd.DataFrame
            The training labels
        weights_train : pd.DataFrame, optional
            The training weights
        epochs : int, optional
            The number of epochs to train the model for, by default 100
        lr : float, optional
            The learning rate, by default 0.001. Lower values will result in
            slower learning, higher values will result in faster learning

        """
        if self.fc1 is None:
            self.setup_model(X_train)

        # check for valid inputs
        if X_train.isnull().any().any():
            raise ValueError("X_train contains null values")
        if y_train.isnull().any():
            raise ValueError("y_train contains null values")
        if np.isinf(y_train).any():
            raise ValueError("y_train contains infinite values")
        if (y_train < 0).any():
            raise ValueError("y_train contains negative values")
        if weights_train is not None and weights_train.isnull().any():
            raise ValueError("weights_train contains null values")

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        optimizer = self.optimizer
        # loss function - mean squared error (MSE)
        self.criterion = nn.MSELoss(reduction="none") if weights_train else nn.MSELoss()
        criterion = self.criterion

        # convert dataframes to tensors
        X_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        if weights_train:
            weights_tensor = torch.tensor(weights_train.to_numpy(), dtype=torch.float32)
            with_weights = "`with weights`,"
        else:
            with_weights = None

        logger.info(
            f"training model {with_weights} \n"
            f"epochs: `{epochs}`,\n"
            f"optimizer: `{type(optimizer).__name__}`,\n"
            f"criterion: `{type(criterion).__name__}`,\n"
            f"learning rate: `{lr}`"
        )

        # training
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X_tensor)
            # compute loss
            if weights_train:
                # weighted
                losses = criterion(outputs.squeeze(), y_tensor)
                weighted_loss = (losses * weights_tensor).mean()
                loss = weighted_loss
            else:
                # unweighted
                loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict the mortality rates.

        Parameters
        ----------
        X_test : pd.DataFrame
            The test data

        Returns
        -------
        np.ndarray
            The predicted mortality rates

        """
        # convert to numpy array
        X_test = X_test.to_numpy()

        self.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            predictions_tensor = self(X_tensor)

        predictions = predictions_tensor.squeeze().numpy()

        return predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function to be called from nn.Module.

        The nn.Module will call this function when there are predictions.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor

        """
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.output(x)

        # non-negative activation
        x = nn.Softplus()(x)

        return x
