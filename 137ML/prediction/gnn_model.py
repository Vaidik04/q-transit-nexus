"""
gnn_model.py — Spatio-Temporal Graph Neural Network (ST-GNN) predictor.

Architecture
------------
A deliberately minimal, defensible architecture for a hackathon timeline:
    1. GCN (Graph Convolutional Network) layers for spatial feature mixing across
       adjacent road edges.
    2. GRU (Gated Recurrent Unit) for temporal modelling over a sequence of past
       time-steps per node.
    3. Linear output head projecting to one value per prediction horizon.

Why this architecture?
  * GCN is simpler and faster to train than GAT/GraphSAGE with comparable
    accuracy on road graphs.
  * GRU captures temporal autocorrelation better than pure sliding-window
    approaches and is far cheaper to train than Transformer-based temporal layers.
  * The full model has ~100K parameters at the default config, fitting well within
    a hackathon training budget.

Graph construction
------------------
Nodes = road edges (segments), indexed by their position in edge_id order.
Spatial adjacency = provided via config or defaulted to a ring topology for
demo purposes.  When SUMO network data is available, replace _build_demo_adjacency()
with a proper SUMO-net parser.

Implements the SAME interface as BaselinePredictor (fit/predict/save/load)
so evaluator.py can swap between them transparently.

Dependencies: PyTorch + PyTorch Geometric (PyG).  If PyG is not installed,
import fails cleanly — evaluator.py catches this and skips GNN evaluation.
"""

from __future__ import annotations

import importlib
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prediction import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch / PyG availability — resolved lazily via importlib so the IDE's
# static analyser never tries to resolve these at parse time.
# Both flags are set at module load; the actual objects are fetched on demand.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    # These imports are only for type-checker / IDE autocomplete.
    # They are NEVER executed at runtime when the packages are absent.
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

def _try_torch():
    """Return (torch, nn, DataLoader, TensorDataset) or raise ImportError."""
    _t = importlib.import_module("torch")
    _nn = importlib.import_module("torch.nn")
    _tud = importlib.import_module("torch.utils.data")
    return _t, _nn, _tud.DataLoader, _tud.TensorDataset

def _try_pyg():
    """Return GCNConv class or raise ImportError."""
    _pyg_nn = importlib.import_module("torch_geometric.nn")
    return _pyg_nn.GCNConv

try:
    torch, nn, DataLoader, TensorDataset = _try_torch()
    _TORCH_AVAILABLE = True
except ImportError as _e:
    _TORCH_AVAILABLE = False
    torch = nn = DataLoader = TensorDataset = None  # type: ignore[assignment]
    logger.warning("PyTorch not available — STGNNPredictor will not function. (%s)", _e)

try:
    GCNConv = _try_pyg()
    _PYG_AVAILABLE = True
except ImportError as _e:
    _PYG_AVAILABLE = False
    GCNConv = None  # type: ignore[assignment,misc]
    logger.warning("PyTorch Geometric not available. (%s)", _e)



# ---------------------------------------------------------------------------
# Neural network definition
# Only defined when both PyTorch and PyG are present.  The `if` guard prevents
# the IDE's static analyser from evaluating nn.Module / GCNConv as base
# classes when those packages are not installed.
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE and _PYG_AVAILABLE:

    class _STGNNModel(nn.Module):  # type: ignore[misc]
        """Spatio-Temporal GNN: GCN spatial layers + GRU temporal layer + linear head.

        Parameters
        ----------
        in_features    : number of input features per node per time-step
        hidden_dim     : GCN and GRU hidden size
        num_gnn_layers : number of GCN layers stacked
        gru_hidden     : GRU hidden size (can differ from GCN hidden)
        num_horizons   : number of output prediction horizons
        dropout        : dropout probability applied after each GCN layer
        seq_len        : number of past time-steps in the input sequence
        """

        def __init__(
            self,
            in_features: int,
            hidden_dim: int,
            num_gnn_layers: int,
            gru_hidden: int,
            num_horizons: int,
            dropout: float,
            seq_len: int,
        ) -> None:
            super().__init__()
            self.seq_len = seq_len
            self.num_horizons = num_horizons

            # GCN layers (spatial)
            self.gcn_layers = nn.ModuleList()
            self.gcn_norms = nn.ModuleList()
            in_dim = in_features
            for _ in range(num_gnn_layers):
                self.gcn_layers.append(GCNConv(in_dim, hidden_dim))
                self.gcn_norms.append(nn.LayerNorm(hidden_dim))
                in_dim = hidden_dim
            self.dropout = nn.Dropout(dropout)

            # GRU temporal layer
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
            )

            # Output head: one value per horizon
            self.head = nn.Linear(gru_hidden, num_horizons)

        def forward(
            self,
            x_seq: "torch.Tensor",
            edge_index: "torch.Tensor",
            num_nodes: int,
        ) -> "torch.Tensor":
            """Forward pass.  Returns tensor of shape [batch*nodes, num_horizons]."""
            BN, T, F = x_seq.shape

            gcn_outputs = []
            for t in range(T):
                x_t = x_seq[:, t, :]
                for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
                    x_t = gcn(x_t, edge_index)
                    x_t = norm(x_t)
                    x_t = torch.relu(x_t)
                    x_t = self.dropout(x_t)
                gcn_outputs.append(x_t.unsqueeze(1))

            gcn_seq = torch.cat(gcn_outputs, dim=1)
            _, h_n = self.gru(gcn_seq)
            h_last = h_n.squeeze(0)
            return self.head(h_last)

else:
    # Placeholder keeps _STGNNModel always defined so STGNNPredictor can
    # reference it without a NameError when PyG is absent.
    _STGNNModel = None  # type: ignore[assignment,misc]




# ---------------------------------------------------------------------------
# STGNNPredictor (public API — same interface as BaselinePredictor)
# ---------------------------------------------------------------------------

class STGNNPredictor:
    """ST-GNN predictor implementing the same fit/predict/save/load interface.

    Parameters
    ----------
    horizons   : prediction horizons (minutes)
    adj_list   : adjacency list {edge_id: [neighbour_edge_ids]}.
                 If None, a ring/chain topology is built from the training nodes.
    """

    def __init__(
        self,
        horizons: List[int] = None,
        adj_list: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if not _TORCH_AVAILABLE or not _PYG_AVAILABLE:
            raise ImportError(
                "STGNNPredictor requires PyTorch and PyTorch Geometric. "
                "Install with: pip install torch torch-geometric"
            )

        self.horizons = horizons or cfg.HORIZONS
        self.adj_list = adj_list
        self.hp = cfg.GNN_HYPERPARAMS
        self._model: Optional["_STGNNModel"] = None
        self._node_index: Optional[Dict[str, int]] = None  # edge_id → node index
        self._edge_index: Optional["torch.Tensor"] = None
        self._feature_names: Optional[List[str]] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("STGNNPredictor initialised on device: %s", self._device)

    # ------------------------------------------------------------------
    # Predictor interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "STGNNPredictor":
        """Train the ST-GNN on the feature/target matrices.

        NOTE: X/Y from evaluator.py are flat (not time-sequenced) because they come
        from the same tabular pipeline as the baseline.  We reconstruct sequences
        here using the last seq_len rows per node group.
        """
        self._feature_names = list(X.columns)
        num_features = len(self._feature_names)
        seq_len = self.hp["sequence_length"]
        hidden_dim = self.hp["hidden_dim"]
        gru_hidden = self.hp["temporal_hidden"]
        num_gnn_layers = self.hp["gnn_layers"]
        dropout = self.hp["dropout"]
        lr = self.hp["lr"]
        epochs = self.hp["epochs"]
        batch_size = self.hp["batch_size"]

        # Attach edge_id temporarily if not in X — we need it for graph structure
        # Evaluator passes X without edge_id; we use row order (nodes round-robin)
        # In a production setup, edge_id would be passed explicitly.
        n_rows = len(X)
        num_nodes = max(1, n_rows // seq_len)  # rough estimate
        self._node_index = {f"node_{i}": i for i in range(num_nodes)}

        # Build adjacency
        edge_index = self._build_edge_index(num_nodes)
        self._edge_index = edge_index.to(self._device)

        # Build sequence tensors: [samples, seq_len, num_features]
        X_arr = X.values.astype(np.float32)
        Y_arr = np.stack(
            [Y[f"target_t{h}"].values.astype(np.float32) for h in self.horizons],
            axis=1,
        )

        X_seq, Y_seq = self._make_sequences(X_arr, Y_arr, seq_len, num_nodes)

        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(Y_seq, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build model
        self._model = _STGNNModel(
            in_features=num_features,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gru_hidden=gru_hidden,
            num_horizons=len(self.horizons),
            dropout=dropout,
            seq_len=seq_len,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                # x_batch: [B*num_nodes, seq_len, F]
                # y_batch: [B*num_nodes, num_horizons]
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                # Repeat edge_index to cover batch size (PyG global index assumption)
                preds = self._model(x_batch, self._edge_index, num_nodes)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d — loss: %.4f", epoch + 1, epochs, epoch_loss / len(loader)
                )

        self._model.eval()
        logger.info("STGNNPredictor training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Run inference — returns {horizon: ndarray}."""
        if self._model is None:
            raise RuntimeError("STGNNPredictor.predict() called before fit().")

        seq_len = self.hp["sequence_length"]
        num_nodes = max(1, len(X) // seq_len)
        X_arr = X.values.astype(np.float32)
        # Dummy Y just for sequence construction shape
        Y_dummy = np.zeros((len(X), len(self.horizons)), dtype=np.float32)
        X_seq, _ = self._make_sequences(X_arr, Y_dummy, seq_len, num_nodes)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self._device)

        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_tensor, self._edge_index, num_nodes)
            preds_np = preds.cpu().numpy()  # [B*N, num_horizons]

        return {
            h: preds_np[:, i] for i, h in enumerate(self.horizons)
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict() if self._model else None,
                "node_index": self._node_index,
                "edge_index": self._edge_index,
                "feature_names": self._feature_names,
                "horizons": self.horizons,
                "hp": self.hp,
                "adj_list": self.adj_list,
            },
            path,
        )
        logger.info("STGNNPredictor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "STGNNPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved GNN model at: {path}")
        state = torch.load(path, map_location="cpu")
        obj = cls(horizons=state["horizons"], adj_list=state.get("adj_list"))
        obj._node_index = state["node_index"]
        obj._edge_index = state["edge_index"]
        obj._feature_names = state["feature_names"]
        obj.hp = state["hp"]

        if state["model_state"] is not None:
            num_nodes = len(obj._node_index) if obj._node_index else 1
            num_features = len(obj._feature_names) if obj._feature_names else 1
            obj._model = _STGNNModel(
                in_features=num_features,
                hidden_dim=obj.hp["hidden_dim"],
                num_gnn_layers=obj.hp["gnn_layers"],
                gru_hidden=obj.hp["temporal_hidden"],
                num_horizons=len(obj.horizons),
                dropout=obj.hp["dropout"],
                seq_len=obj.hp["sequence_length"],
            )
            obj._model.load_state_dict(state["model_state"])
            obj._model.eval()
        logger.info("STGNNPredictor loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_edge_index(self, num_nodes: int) -> "torch.Tensor":
        """Build a simple ring graph adjacency for demo purposes.

        Replace with SUMO network adjacency when available.
        """
        if self.adj_list is not None:
            # Build from provided adjacency list
            src, dst = [], []
            node_to_idx = {k: i for i, k in enumerate(self.adj_list.keys())}
            for src_node, neighbours in self.adj_list.items():
                for dst_node in neighbours:
                    if dst_node in node_to_idx:
                        src.append(node_to_idx[src_node])
                        dst.append(node_to_idx[dst_node])
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            # Ring topology: i → i+1, last → 0 (undirected: both directions)
            src = list(range(num_nodes)) + list(range(1, num_nodes)) + [0]
            dst = list(range(1, num_nodes)) + [0] + list(range(num_nodes))
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index

    @staticmethod
    def _make_sequences(
        X: np.ndarray,
        Y: np.ndarray,
        seq_len: int,
        num_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape flat [N, F] array into sequences [samples*nodes, seq_len, F].

        We treat every seq_len consecutive rows (for each 'virtual' node) as one
        sequence sample.  This is a simplified approach — a production version would
        use actual edge_id groups and proper sliding windows.
        """
        n_samples = len(X) // (seq_len * num_nodes)
        if n_samples == 0:
            n_samples = 1
            # Pad with zeros if we have fewer rows than seq_len * num_nodes
            pad_len = seq_len * num_nodes - len(X)
            X = np.vstack([np.zeros((pad_len, X.shape[1]), dtype=np.float32), X])
            Y = np.vstack([np.zeros((pad_len, Y.shape[1]), dtype=np.float32), Y])

        total = n_samples * seq_len * num_nodes
        X_trimmed = X[:total]
        Y_trimmed = Y[:total]

        # [n_samples * num_nodes, seq_len, F]
        X_seq = X_trimmed.reshape(n_samples * num_nodes, seq_len, X.shape[1])
        # [n_samples * num_nodes, num_horizons]
        Y_seq = Y_trimmed.reshape(n_samples * num_nodes, seq_len, Y.shape[1])[:, -1, :]

        return X_seq, Y_seq
