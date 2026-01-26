"""Creates neural models for forecasting mortality rates."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
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
    The model uses a hybrid architecture with two parallel paths that are summed
    to produce the final output:

    1. Wide Path (Linear/GLM):
       - Captures direct linear relationships and the baseline rate.
       - Architecture: input -> wide_linear -> output

    The layers are:
        - wide_linear: input_size -> 1

    2. Deep Path (Non-linear interactions):
       - Captures complex interactions and residuals.
       - Architecture: fc1 -> relu1 -> fc2 -> relu2 -> fc3 -> relu3 -> deep_output

    The layers are:
        - fc1: input_size -> 32
        - fc2: 32 -> 32
        - fc3: 32 -> 16
        - deep_output: 16 -> 1

    prediction = wide_linear(x) + deep_output(x)

    """

    def __init__(
        self,
        task: str = "poisson",
        embedding_cols: Optional[list] = None,
        embedding_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        task : str, optional
            Either "poisson" or "binomial"
        embedding_cols : list, optional
            Categorical columns to create embeddings for
        embedding_dims : dict, optional
            Dictionary mapping categorical feature names to their embedding dimensions
            e.g., {"age_group": 8, "region": 4}
            If None, will use min(50, (vocab_size + 1) // 2) for each feature

        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.feature_names = None
        if embedding_cols is None:
            embedding_cols = []
        self.embedding_cols = embedding_cols
        self.num_cols = []
        if embedding_dims is None:
            embedding_dims = {}
        self.embedding_dims = embedding_dims
        self.embeddings = nn.ModuleDict()
        self.wide_linear = None
        self.fc1 = self.fc2 = self.fc3 = self.output = None
        self.relu1 = self.relu2 = self.relu3 = None
        self.label_encoders = {}
        self._is_fitted = False
        self.to(self.device)
        logger.info(
            f"initialized Neural model with Torch\n"
            f"task: {self.task} \n"
            f"device: {self.device}"
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
        num_cols = [col for col in X_train.columns if col not in self.embedding_cols]
        self.feature_names = X_train.columns
        self.num_cols = num_cols
        logger.info(f"non-embedding columns: {self.num_cols}")
        logger.info(f"embedding columns: {self.embedding_cols}")

        input_size = len(num_cols)
        total_embedding_dim = self._create_embeddings(X_train)
        input_size += total_embedding_dim

        # create layers
        # wide
        self.wide_linear = nn.Linear(input_size, 1)

        # deep
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
        with torch.no_grad():
            self.output.weight.fill_(0.0)
            self.output.bias.fill_(0.0)

    def forward(
        self, X_torch_num: torch.Tensor, X_torch_embed_idx: [torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward function to be called from nn.Module.

        The nn.Module will call this function when there are predictions.

        Parameters
        ----------
        X_torch_num : torch.Tensor
            Numeric features
        X_torch_embed_idx : list
            Index of embedding features

        Returns
        -------
        torch.Tensor
            The output tensor

        """
        if self.embedding_cols:
            embedding_vectors = [
                self.embeddings[col](idx)
                for col, idx in zip(self.embedding_cols, X_torch_embed_idx, strict=True)
            ]
            x = torch.cat(embedding_vectors, dim=1)
        else:
            # no embedding columns, make a zero tensor
            n = X_torch_num.size(0) if X_torch_num is not None else 0
            x = torch.zeros(
                (n, 0),
                dtype=torch.float32,
                device=X_torch_num.device if X_torch_num is not None else None,
            )

        # if numeric features are present, combine with embeddings
        if X_torch_num is not None:
            x = torch.cat([x, X_torch_num], dim=1)

        # copy input for wide linear
        x_clone = x.clone()

        # forward pass
        x_wide = self.wide_linear(x_clone).squeeze(-1)

        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)
        x_deep = self.output(x).squeeze(-1)

        return x_wide + x_deep

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        weights_test: pd.Series,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: Optional[int] = None,
        dropout: float = 0.0,
        weight_decay: float = 0.0001,
        early_stopping: bool = True,
        warmup_epochs: int = 50,
        max_patience: int = 20,
        seed: Optional[int] = None,
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
        X_test : pd.DataFrame
            The testing data
        y_test : pd.Series
            The testing labels
        weights_test : pd.Series
            The weights for the testing data
        epochs : int, optional (default=100)
            The number of epochs to train the model for, by default 100
        lr : float, optional (default=0.001)
            The learning rate, by default 0.001. Lower values will result in
        batch_size: int, optional (default=None)
            The batch size for mini-batch training. If None, use full-batch training
            slower learning, higher values will result in faster learning
        dropout : float, optional (default=0.0)
            Dropout rate for the model
        weight_decay : float, optional (default=1e-4)
            similar to L2 regularization to prevent overfitting
        early_stopping : bool, optional (default=True)
            Whether to use early stopping based on test loss
        warmup_epochs : int, optional (default=50)
            Number of epochs to wait before starting early stopping
        max_patience : int, optional (default=20)
            Number of epochs with no improvement to wait before stopping training
        seed : int, optional
            Random seed for reproducibility

        """
        # defaults
        MAX_NORM = 5.0
        MAX_NEGATIVE_LOG = -30.0
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0

        # set seed for weight initialization and dropout masks
        if seed is not None:
            logger.info(f"setting seed: `{seed}`")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # validations
        if self.fc1 is None:
            self.setup_model(X_train=X, dropout=dropout)
            self.to(self.device)
        else:
            logger.warning(
                "model has already been set up and calling fit again"
                "will update existing weights."
            )
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
        y_test = y_test * weights_test

        # convert to torch tensors
        X_torch_num, X_torch_embed_idx = self._prepare_input_tensor(X)
        X_torch_test_num, X_torch_test_embed_idx = self._prepare_input_tensor(X_test)

        y_torch = torch.tensor(
            y.to_numpy().reshape(-1), dtype=torch.float32, device=self.device
        )
        y_torch_length = len(y_torch)
        y_torch_test = torch.tensor(
            y_test.to_numpy().reshape(-1), dtype=torch.float32, device=self.device
        )
        weights_torch = torch.tensor(
            weights.to_numpy().reshape(-1), dtype=torch.float32, device=self.device
        )
        weights_torch_test = torch.tensor(
            weights_test.to_numpy().reshape(-1), dtype=torch.float32, device=self.device
        )

        if X_torch_embed_idx:
            X_torch_embed_stacked = torch.stack(X_torch_embed_idx, dim=1)
        else:
            X_torch_embed_stacked = torch.zeros(
                y_torch_length, 0, dtype=torch.long, device=self.device
            )

        # batch configuration
        # if batch_size is None or >= dataset size, use full-batch
        use_mini_batch = batch_size is not None and batch_size < y_torch_length

        train_losses, test_losses, learning_rates = [], [], []

        # setup optimizer and a learning rate scheduler to reduce learning rate
        # when loss plateaus
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10
        )

        # initialize prediction to global rate
        overall_mu = float(y.sum() / weights.sum())
        logger.info(f"overall_mu: {overall_mu:.6f}")
        with torch.no_grad():
            self.wide_linear.bias.fill_(
                np.log(max(overall_mu, 1e-12)).astype(np.float32)
            )
            self.output.bias.fill_(0.0)
            self.output.weight.mul_(0.01)

        # logging
        logger.info(f"epochs: {epochs:,.0f}, batch_size: {batch_size}, lr: {lr}")
        logger.info(f"dropout: {dropout}, weight_decay: {weight_decay}")
        if early_stopping:
            logger.info(
                f"early stopping: enabled, warmup_epochs: {warmup_epochs}, "
                f"max_patience: {max_patience}"
            )
        else:
            logger.info("early stopping: disabled")

        # train with loss likelihoods
        pbar = tqdm(range(epochs), desc="Training", leave=True)
        for epoch in pbar:
            # training forward pass
            self.train()
            epoch_loss = 0.0
            num_batches = 0

            if use_mini_batch:
                # shuffle
                perm = torch.randperm(y_torch_length, device=self.device)

                # iterate over batches using direct indexing
                # instead of dataloader for efficiency
                for start_idx in range(0, y_torch_length, batch_size):
                    end_idx = min(start_idx + batch_size, y_torch_length)
                    idx = perm[start_idx:end_idx]

                    batch_num = X_torch_num[idx]
                    batch_embed = X_torch_embed_stacked[idx]
                    batch_y = y_torch[idx]
                    batch_weights = weights_torch[idx]

                    opt.zero_grad()

                    batch_embed_list = [
                        batch_embed[:, i] for i in range(batch_embed.shape[1])
                    ]

                    z_torch = self(batch_num, batch_embed_list)

                    if self.task == "poisson":
                        logE = torch.log(batch_weights).clamp(min=MAX_NEGATIVE_LOG)
                        loglam = z_torch + logE
                        loss = F.poisson_nll_loss(
                            input=loglam,
                            target=batch_y,
                            log_input=True,
                            full=False,
                            reduction="mean",
                        )
                    else:  # binomial
                        loss = -(
                            batch_y * F.logsigmoid(z_torch)
                            + (batch_weights - batch_y) * F.logsigmoid(-z_torch)
                        ).mean()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=MAX_NORM)
                    opt.step()

                    epoch_loss += loss.detach().item()
                    num_batches += 1

            # full-batch training
            else:
                opt.zero_grad()

                X_torch_embed_list = [
                    X_torch_embed_stacked[:, i]
                    for i in range(X_torch_embed_stacked.shape[1])
                ]

                z_torch = self(X_torch_num, X_torch_embed_list)

                if self.task == "poisson":
                    logE = torch.log(weights_torch).clamp(min=MAX_NEGATIVE_LOG)
                    loglam = z_torch + logE
                    loss = F.poisson_nll_loss(
                        input=loglam,
                        target=y_torch,
                        log_input=True,
                        full=False,
                        reduction="mean",
                    )
                else:  # binomial
                    loss = -(
                        y_torch * F.logsigmoid(z_torch)
                        + (weights_torch - y_torch) * F.logsigmoid(-z_torch)
                    ).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=MAX_NORM)
                opt.step()

                epoch_loss = loss.detach().item()
                num_batches = 1

            train_loss_value = epoch_loss / num_batches

            # test loss - no gradients needed
            with torch.no_grad():
                self.eval()
                z_torch_test = self(X_torch_test_num, X_torch_test_embed_idx)
                if self.task == "poisson":
                    logE_test = torch.log(weights_torch_test).clamp(
                        min=MAX_NEGATIVE_LOG
                    )
                    loglam_test = z_torch_test + logE_test
                    test_loss = F.poisson_nll_loss(
                        input=loglam_test,
                        target=y_torch_test,
                        log_input=True,
                        full=False,
                        reduction="mean",
                    )
                else:
                    test_loss = -(
                        y_torch_test * F.logsigmoid(z_torch_test)
                        + (weights_torch_test - y_torch_test)
                        * F.logsigmoid(-z_torch_test)
                    ).mean()

            test_loss_value = test_loss.item()
            train_losses.append(train_loss_value)
            test_losses.append(test_loss_value)
            learning_rates.append(opt.param_groups[0]["lr"])

            scheduler.step(test_loss_value)

            # early stopping when loss does not improve
            if early_stopping:
                if test_loss_value < best_loss:
                    best_loss = test_loss_value
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = self.state_dict().copy()
                elif epoch >= warmup_epochs:
                    patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(
                        f"early stopping at epoch: {epoch + 1}, "
                        f"best epoch: {best_epoch}"
                    )
                    pbar.close()
                    break

            pbar.set_postfix(
                {
                    "train": f"{train_loss_value:,.2f}",
                    "test": f"{test_loss_value:,.2f}",
                    "counter": patience_counter,
                    "lr": opt.param_groups[0]["lr"],
                }
            )

        # load best state
        if early_stopping and best_state is not None:
            self.load_state_dict(best_state)
            logger.info(
                f"training complete, restored best epoch: `{best_epoch}` "
                f"with test loss: `{best_loss:.6f}`"
            )
        else:
            logger.info(
                f"training complete, {epoch + 1} epochs, "
                f"with test loss: `{test_loss_value:,.2f}`"
            )
        self._is_fitted = True

        # create loss plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=train_losses,
                mode="lines+markers",
                name="Train Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=test_losses,
                mode="lines+markers",
                name="Test Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=learning_rates,
                mode="lines",
                name="Learning Rate",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Neural Network Training",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2={
                "title": "Learning Rate",
                "overlaying": "y",
                "side": "right",
            },
        )
        return fig

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict the target.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The features
            If np.ndarray, must have the same column order as training data
            and embedding columns contain encoded integer indices.

        Returns
        -------
        predictions : np.ndarray
            The predictions

        """
        # initialize
        embedding_cols_mapped_back = []

        # convert array to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # ensure non-embedding columns are numeric
        for col in self.num_cols:
            if col in X.columns and X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # map embedding columns back to strings if needed
        for col in self.embedding_cols:
            # check string
            if X[col].dtype == "object" or isinstance(X[col].iloc[0], str):
                valid_values = set(self.label_encoders[col].keys())
                X[col] = X[col].apply(
                    lambda x, valid_values=valid_values: x
                    if x in valid_values
                    else "__UNK__"
                )
            # map integers back to strings
            else:
                embedding_cols_mapped_back.append(col)
                reverse_encoder = {v: k for k, v in self.label_encoders[col].items()}
                max_idx = max(self.label_encoders[col].values())
                indices = X[col].astype(float).round().astype(int).clip(0, max_idx)
                X[col] = indices.map(reverse_encoder).fillna("__UNK__")

        if embedding_cols_mapped_back:
            logger.warning(
                f"mapped embedding columns `{embedding_cols_mapped_back}` "
                f"from integer indices back to strings for prediction"
            )

        # make prediction
        self.eval()
        X_torch_num, X_torch_embed_idx = self._prepare_input_tensor(X)
        with torch.no_grad():
            z_torch = self(X_torch_num, X_torch_embed_idx).cpu().numpy()

        # convert to rate
        if self.task == "poisson":
            mu = np.exp(z_torch)
            q = mu
            predictions = np.clip(q, 1e-9, 1 - 1e-9)

        else:  # binomial
            q = 1.0 / (1.0 + np.exp(-z_torch))
            predictions = np.clip(q, 1e-9, 1 - 1e-9)

        return predictions

    def embedding_get_weights(self, embed_col: str) -> pd.DataFrame:
        """
        Get embedding weights for a embedding column.

        Parameters
        ----------
        embed_col : str
            The embedding column name

        Returns
        -------
        weights_df : pd.DataFrame
            DataFrame with embedding name as index and embedding dimensions as columns

        """
        if embed_col not in self.embeddings:
            raise ValueError(f"No embedding found for '{embed_col}'")

        # get the weights
        weights = self.embeddings[embed_col].weight.detach().cpu().numpy()
        idx_to_label = {v: k for k, v in self.label_encoders[embed_col].items()}
        labels = [idx_to_label.get(i, f"idx_{i}") for i in range(weights.shape[0])]
        weights_df = pd.DataFrame(
            weights, index=labels, columns=[f"dim_{i}" for i in range(weights.shape[1])]
        )
        weights_df = weights_df.sort_index()

        return weights_df

    def embedding_plot_similarity(self, embed_col: str) -> go.Figure:
        """
        Plot a heatmap of cosine similarities between category embeddings.

        This is useful to understand similarities between 2 categories.

        Parameters
        ----------
        embed_col : str
            The embedding column name

        Returns
        -------
        similarity_fig : Figure
            Heatmap figure of cosine similarities

        """
        # get weights
        weights_df = self.embedding_get_weights(embed_col)
        weights_df = weights_df.drop("__UNK__", errors="ignore")
        weights_df = weights_df.sort_index()

        # compute cosine similarity
        sim_matrix = cosine_similarity(weights_df.values)
        sim_df = pd.DataFrame(
            sim_matrix, index=weights_df.index, columns=weights_df.index
        )

        # create truncated labels
        truncated_labels = [str(label)[:5] for label in sim_df.index]

        # plot
        similarity_fig = px.imshow(
            sim_df,
            title=f"Embedding Similarity: {embed_col}",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )

        # update axes with truncated labels (full names still in hover)
        similarity_fig.update_xaxes(
            ticktext=truncated_labels,
            tickvals=list(range(len(truncated_labels))),
        )
        similarity_fig.update_yaxes(
            ticktext=truncated_labels,
            tickvals=list(range(len(truncated_labels))),
        )

        return similarity_fig

    def embedding_plot_2d(self, embed_col: str, method: str = "tsne") -> go.Figure:
        """
        Plot embeddings in 2D using PCA or t-SNE.

        This is useful to visualize the embedding space.
        - t-SNE is good for visualizing clusters. The closer the points are,
            the more similar the categories are.

        Parameters
        ----------
        embed_col : str
            The embedding column name
        method : str, optional (default='tsne')
            'pca' or 'tsne'

        Returns
        -------
        embedding_fig : Figure
            2D scatter plot of embeddings

        """
        # get weights
        weights_df = self.embedding_get_weights(embed_col)
        weights_df = weights_df.drop("__UNK__", errors="ignore")
        if weights_df.shape[1] < 2:
            raise ValueError("Need at least 2 embedding dimensions for 2D plot")

        # reduce dimensions
        if method == "pca":
            reducer = PCA(n_components=2)
        else:
            perplexity = min(30, max(5, len(weights_df) - 1))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)

        coords = reducer.fit_transform(weights_df.values)

        plot_df = pd.DataFrame(
            {"x": coords[:, 0], "y": coords[:, 1], "category": weights_df.index}
        )

        # plot
        embedding_fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            text="category",
            title=f"{embed_col} Embeddings ({method.upper()})",
        )
        embedding_fig.update_traces(textposition="top center")

        return embedding_fig

    def _create_embeddings(self, X: pd.DataFrame) -> int:
        """
        Create embeddings for categorical features.

        When to use embeddings is important to think about. Embeddings are
        useful when there are high-cardinatility categorical features with
        complex relationships (e.g. >10). If there are only a few categories,
        a different encoding may be more appropriate.

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

        for embed_feature in self.embedding_cols:
            # create label encoder
            unique_values = X[embed_feature].dropna().unique()
            self.label_encoders[embed_feature] = {
                "__UNK__": 0,
                **{val: idx + 1 for idx, val in enumerate(unique_values)},
            }
            vocab_size = len(self.label_encoders[embed_feature])

            # warn if embeddings may not be appropriate (e.g. <=10 unique values)
            if vocab_size <= 3:
                raise ValueError(
                    f"embedding feature '{embed_feature}' has only 2 unique values "
                    f"and not suitable for embedding; consider ordinal or "
                    f"one-hot encoding instead"
                )
            elif vocab_size <= 11:
                logger.warning(
                    f"embedding feature '{embed_feature}' has only {vocab_size - 1} "
                    f"unique values and may be better suited for one-hot encoding"
                )

            if embed_feature not in self.embedding_dims:
                # use a rule of thumb for embedding dimensions
                # capping at 50, and generally half the vocabulary size
                self.embedding_dims[embed_feature] = min(50, (vocab_size + 1) // 2)

            embed_dim = self.embedding_dims[embed_feature]
            self.embeddings[embed_feature] = nn.Embedding(vocab_size, embed_dim).to(
                self.device
            )
            total_embedding_dim += embed_dim

            # initialize embeddings
            nn.init.xavier_uniform_(self.embeddings[embed_feature].weight)

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
        X_torch_embed_idx : list
            Index of Categorical features

        """
        X_torch_num = None

        # numeric features
        if self.num_cols:
            X_torch_num = torch.tensor(
                X[self.num_cols].to_numpy(), dtype=torch.float32, device=self.device
            )

        # embedding features
        X_torch_embed_idx = []
        for embed_col in self.embedding_cols:
            mapped_values = X[embed_col].map(self.label_encoders[embed_col])
            # handle missing values by adding 0 to categories if needed
            if (
                isinstance(mapped_values.dtype, pd.CategoricalDtype)
                and 0 not in mapped_values.cat.categories
            ):
                mapped_values = mapped_values.cat.add_categories([0])

            # label encode
            idx = mapped_values.fillna(0).astype("int64").to_numpy()
            X_torch_embed_idx.append(torch.from_numpy(idx).to(self.device))

        return X_torch_num, X_torch_embed_idx


class Shap:
    """
    A wrapper class for SHAP explainability for the Neural model.

    The wrapper may be expanded in the future to include other explainability methods.

    Higher SHAP values increase the prediction, lower SHAP values decrease the
    prediction. Magnitude of SHAP values indicates the strength of the feature's effect.

    Usage:
    --------------
    Shap = Shap(
        model=neural_model,
        background_df=X_train,
        n_samples=100,
        seed=42,
    )
    shap_values = Shap.compute_values(
        explain_df=X_test,
        n_samples=100,
        seed=42,
    )
    """

    def __init__(
        self,
        model: Neural,
        background_df: pd.DataFrame,
        n_samples: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the SHAP wrapper.

        Parameters
        ----------
        model : Neural
            The trained Neural model
        background_df : pd.DataFrame
            The background_df data to use as background for SHAP
            Typically this is the training data
        n_samples : int, optional (default=100)
            Number of background samples to use
        seed : int, optional
            Random seed for reproducibility when sampling background data

        """
        self.model = model
        self.n_samples = n_samples
        self.seed = seed

        self.sample_background_df = None
        self.sample_explain_df = None
        self.shap_values = None

        self.explainer = self._create_explainer(background_df=background_df)

    def _create_explainer(self, background_df: pd.DataFrame) -> shap.KernelExplainer:
        """
        Create a SHAP KernelExplainer for the Neural model.

        The KernelExplainer is model-agnostic and works well with embeddings by
        treating the model as a black box. It uses a background dataset to
        compute expected values.

        Parameters
        ----------
        background_df : pd.DataFrame
            The background_df data to use as background for SHAP
            Typically this is the training data

        Returns
        -------
        explainer : shap.KernelExplainer
            An explainer object for computing SHAP values

        """
        # initiate variables
        model = self.model
        n_samples = self.n_samples
        seed = self.seed

        # validations
        if model.fc1 is None:
            raise ValueError("Model must be fitted before creating explainer")

        # sample background data
        sample_background_df = background_df.sample(
            n=n_samples, random_state=seed
        ).copy()

        # create explainer
        seed_str = f" and a seed of `{seed}`" if seed is not None else ""
        logger.info(
            f"creating SHAP KernelExplainer with `{n_samples}` "
            f"background samples{seed_str}"
        )

        explainer = shap.KernelExplainer(model.predict, sample_background_df)

        # save variables
        self.sample_background_df = sample_background_df

        return explainer

    def compute_values(
        self,
        explain_df: pd.DataFrame,
        n_samples: int = 100,
        seed: Optional[int] = None,
    ) -> shap.Explanation:
        """
        Compute SHAP values for a dataset.

        Parameters
        ----------
        explain_df : pd.DataFrame
            Data to explain
            Typically this is a different slice of training data or testing data
        n_samples : int, optional (default=100)
            Number of samples to explain. If None, explains all rows.
            For large datasets, consider using a subset.
        seed : int, optional
            Random seed for reproducibility when sampling

        Returns
        -------
        shap_values : shap.Explanation
            An explanation object containing SHAP values to be used for
            analysis and plotting.

        """
        # initiate variables
        explainer = self.explainer

        # sample explain data
        if n_samples is None:
            sample_explain_df = explain_df.copy()
        else:
            sample_explain_df = explain_df.sample(n=n_samples, random_state=seed).copy()

        # compute shap values and create explanation object
        seed_str = f" and a seed of `{seed}`" if seed is not None else ""
        logger.info(
            f"calculating shap_values with `{n_samples}` explainer samples{seed_str}"
        )

        shap_values = explainer(sample_explain_df)

        # save variables
        self.sample_explain_df = sample_explain_df
        self.shap_values = shap_values

        return shap_values
