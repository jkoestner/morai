"""
Creates neural models for forecasting mortality rates.

The neural network does not perform well on small tabular data. The relationships
it finds do not line up with the relationships in the data.

"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm.auto import tqdm

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
    The training uses loss likelihoods for the loss function based on the task.
    The task can be either "poisson" or "binomial".

    """

    def __init__(
        self,
        task: str = "poisson",
        cat_cols: Optional[list] = None,
        embedding_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        task : str, optional
            Either "poisson" or "binomial"
        cat_cols : list, optional
            Categorical columns
        embedding_dims : dict, optional
            Dictionary mapping categorical feature names to their embedding dimensions
            e.g., {"age_group": 8, "region": 4}
            If None, will use min(50, (vocab_size + 1) // 2) for each feature

        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.feature_names = None
        if cat_cols is None:
            cat_cols = []
        self.cat_cols = cat_cols
        self.num_cols = []
        if embedding_dims is None:
            embedding_dims = {}
        self.embedding_dims = embedding_dims
        self.embeddings = nn.ModuleDict()
        self.fc1 = self.fc2 = self.fc3 = self.output = None
        self.relu1 = self.relu2 = self.relu3 = None
        self.label_encoders = {}
        self.to(self.device)
        logger.info(
            f"initialized Neural model with Torch\n"
            f"task: {self.task} \n"
            f"device: {self.device}"
            f"dropout: {self.dropout}"
        )

    def setup_model(self, X_train: pd.DataFrame, dropout: float = 0.0) -> None:
        """
        Model architecture setup.

        Parameters
        ----------
        X_train : pd.DataFrame
            A DataFrame containing the data to structure.
        dropout : float, optional
            Dropout rate for the model

        """
        # get input size
        num_cols = [col for col in X_train.columns if col not in self.cat_cols]
        self.feature_names = X_train.columns
        self.num_cols = num_cols
        logger.info(f"numeric columns: {self.num_cols}")
        logger.info(f"categorical columns: {self.cat_cols}")

        input_size = len(num_cols)
        total_embedding_dim = self._create_embeddings(X_train)
        input_size += total_embedding_dim

        # create layers
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.output = nn.Linear(16, 1)

    def forward(
        self, X_torch_num: torch.Tensor, X_torch_cat_idx: [torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward function to be called from nn.Module.

        The nn.Module will call this function when there are predictions.

        Parameters
        ----------
        X_torch_num : torch.Tensor
            Numeric features
        X_torch_cat_idx : list
            Index of Categorical features

        Returns
        -------
        torch.Tensor
            The output tensor

        """
        if self.cat_cols:
            embedding_vectors = [
                self.embeddings[col](idx)
                for col, idx in zip(self.cat_cols, X_torch_cat_idx, strict=True)
            ]
            x = torch.cat(embedding_vectors, dim=1)
        else:
            # no categorical columns, make a zero tensor
            n = X_torch_num.size(0) if X_torch_num is not None else 0
            x = torch.zeros(
                (n, 0),
                dtype=torch.float32,
                device=X_torch_num.device if X_torch_num is not None else None,
            )

        # if numeric features are present, combine with embeddings
        if X_torch_num is not None:
            x = torch.cat([x, X_torch_num], dim=1)

        # forward pass
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)
        x = self.output(x).squeeze(-1)

        return x

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series,
        epochs: int = 100,
        lr: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            The training data
        y : pd.Series
            The training labels
        weights : pd.Series
            The weights for the training data
        epochs : int, optional
            The number of epochs to train the model for, by default 100
        lr : float, optional
            The learning rate, by default 0.001. Lower values will result in
            slower learning, higher values will result in faster learning
        dropout : float, optional
            Dropout rate for the model

        """
        # validations
        if self.fc1 is None:
            self.setup_model(X, dropout)
        if self.task not in ("poisson", "binomial"):
            raise ValueError("task must be 'poisson' or 'binomial'")
        if not (X.index.equals(y.index) and X.index.equals(weights.index)):
            raise ValueError("X, y, weights must share the same index")
        bad = (weights <= 0) | weights.isna() | y.isna()
        if bad.any():
            logger.warning(
                f"removing `{bad.sum()}` rows with data that had weights <= 0 or na"
            )
            X = X.loc[~bad]
            y = y.loc[~bad]
            weights = weights.loc[~bad]

        # convert y_train from rate to deaths
        y = y * weights

        # convert to torch tensors
        X_torch_num, X_torch_cat_idx = self._prepare_input_tensor(X)
        y_torch = torch.tensor(
            y.to_numpy().reshape(-1), dtype=torch.float32, device=self.device
        )
        weights_torch = torch.tensor(
            weights.to_numpy().reshape(-1), dtype=torch.float32, device=self.device
        )

        # setup optimizer and a learning rate scheduler to reduce learning rate
        # when loss plateaus
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10
        )

        # initialize prediction to global rate
        overall_mu = float(y.sum() / weights.sum())
        print(f"overall_mu: {overall_mu}")
        with torch.no_grad():
            self.output.bias.fill_(np.log(max(overall_mu, 1e-12)).astype(np.float32))

        # train with loss likelihoods
        best_loss = float("inf")
        patience_counter = 0

        pbar = tqdm(range(epochs), desc="Training", leave=True)
        for epoch in pbar:
            self.train()
            opt.zero_grad()

            # convert to torch tensors, prepare fresh
            z_torch = self(X_torch_num, X_torch_cat_idx)

            if self.task == "poisson":
                logE = torch.log(weights_torch).clamp(min=-30.0)
                loglam = z_torch + logE
                loss = F.poisson_nll_loss(
                    input=loglam,
                    target=y_torch,
                    log_input=True,
                    full=False,
                    reduction="mean",
                )

            else:  # binomial
                logger.warning("binomial model not tested, use with caution")
                loss = -(
                    y_torch * F.logsigmoid(z_torch)
                    + (weights_torch - y_torch) * F.logsigmoid(-z_torch)
                ).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            opt.step()

            scheduler.step(loss)

            # early stopping when loss does not improve for 20 epochs
            loss_value = loss.detach().item()
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 20:
                logger.info(f"early stopping at epoch {epoch + 1}")
                pbar.close()
                break

            pbar.set_postfix({"loss": f"{loss_value:.6f}"})

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict the target.

        Parameters
        ----------
        X : pd.DataFrame
            The features

        Returns
        -------
        predictions : pd.Series
            The predictions

        """
        # make prediction
        self.eval()
        X_torch_num, X_torch_cat_idx = self._prepare_input_tensor(X)
        with torch.no_grad():
            z_torch = self(X_torch_num, X_torch_cat_idx).cpu().numpy()

        # convert to rate
        if self.task == "poisson":
            mu = np.exp(z_torch)
            q = mu
            predictions = pd.Series(np.clip(q, 1e-9, 1 - 1e-9))

        else:  # binomial
            q = 1.0 / (1.0 + np.exp(-z_torch))
            predictions = pd.Series(np.clip(q, 1e-9, 1 - 1e-9))

        return predictions

    def _create_embeddings(self, X: pd.DataFrame) -> int:
        """
        Create embeddings for categorical features.

        Parameters
        ----------
        X : pd.DataFrame
            A DataFrame containing the data to structure.

        Returns
        -------
        total_embedding_dim : int
            The total embedding dimension

        """
        # set up embeddings
        total_embedding_dim = 0

        for cat_feature in self.cat_cols:
            # create label encoder
            unique_values = X[cat_feature].dropna().unique()
            self.label_encoders[cat_feature] = {
                "__UNK__": 0,
                **{val: idx + 1 for idx, val in enumerate(unique_values)},
            }
            vocab_size = len(self.label_encoders[cat_feature])

            if cat_feature not in self.embedding_dims:
                # use a rule of thumb for embedding dimensions
                # capping at 50, and generally half the vocabulary size
                self.embedding_dims[cat_feature] = min(50, (vocab_size + 1) // 2)

            embed_dim = self.embedding_dims[cat_feature]
            self.embeddings[cat_feature] = nn.Embedding(vocab_size, embed_dim)
            total_embedding_dim += embed_dim

            # initialize embeddings
            nn.init.xavier_uniform_(self.embeddings[cat_feature].weight)

        if total_embedding_dim > 0:
            logger.info(f"created embeddings for `{self.embedding_dims}`")

        return total_embedding_dim

    def _prepare_input_tensor(
        self, X: pd.DataFrame
    ) -> Tuple[torch.Tensor, [torch.Tensor]]:
        """
        Prepare input tensor by combining numerical features and embeddings list.

        This will be used as a lookup for the embeddings in the forward pass.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        X_torch_num : torch.Tensor
            Numeric features
        X_torch_cat_idx : list
            Index of Categorical features

        """
        X_torch_num = None

        # numeric features
        if self.num_cols:
            X_torch_num = torch.tensor(
                X[self.num_cols].to_numpy(), dtype=torch.float32, device=self.device
            )

        # categorical features
        X_torch_cat_idx = []
        for cat_col in self.cat_cols:
            mapped_values = X[cat_col].map(self.label_encoders[cat_col])
            # handle missing values by adding 0 to categories if needed
            if (
                isinstance(mapped_values.dtype, pd.CategoricalDtype)
                and 0 not in mapped_values.cat.categories
            ):
                mapped_values = mapped_values.cat.add_categories([0])

            # label encode
            idx = mapped_values.fillna(0).astype("int64").to_numpy()
            X_torch_cat_idx.append(torch.from_numpy(idx).to(self.device))

        return X_torch_num, X_torch_cat_idx
