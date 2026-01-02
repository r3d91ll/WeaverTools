"""
Vendored Atlas model classes for TheLoom interpretability integration.

This module contains the core Atlas model components vendored from:
/home/todd/olympus/models/Atlas/src/model/

Architecture per block (Miras framework):
    Input -> Memory -> Attention -> Gate -> FFN -> Output

Memory and Attention run in parallel, combined via learned gate.
This follows the MAG (Memory as Gating) variant from Titans.

References:
- Titans paper Section 4.2
- "It's All Connected" Section 5
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Retention Gate Module (from retention.py)
# =============================================================================


class RetentionGate(nn.Module):
    """
    Retention gate with local and global components.

    Learning-Retaining objective:
        W_t = argmin_W [loss(W; k_t, v_t) + Ret_t(W, W_{t-1})]

    Where:
        Ret_t = lam_local * ||W - W_{t-1}||_F^2 + lam_global * ||W||_F^2

    The gradient of this retention term is added to the memory update.

    Args:
        d_key: Key dimension
        d_value: Value dimension
        init_local: Initial local retention coefficient (0-1)
        init_global: Initial global retention coefficient (0-1)
        learn_coefficients: Whether to learn retention coefficients
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        init_local: float = 0.5,
        init_global: float = 0.1,
        learn_coefficients: bool = True,
    ):
        super().__init__()

        self.d_key = d_key
        self.d_value = d_value

        if learn_coefficients:
            # Parameterize via sigmoid for (0, 1) range
            # Store pre-sigmoid values
            self.local_logit = nn.Parameter(torch.tensor(init_local).logit())
            self.global_logit = nn.Parameter(torch.tensor(init_global).logit())
        else:
            self.register_buffer("local_logit", torch.tensor(init_local).logit())
            self.register_buffer("global_logit", torch.tensor(init_global).logit())

    @property
    def lambda_local(self) -> torch.Tensor:
        """Local retention coefficient in (0, 1)."""
        return torch.sigmoid(self.local_logit)

    @property
    def lambda_global(self) -> torch.Tensor:
        """Global retention coefficient in (0, 1)."""
        return torch.sigmoid(self.global_logit)

    def compute_penalty_gradient(
        self,
        W: torch.Tensor,
        W_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute gradient of retention penalty w.r.t. W.

        Retention penalty:
            Ret = lam_local * ||W - W_prev||_F^2 + lam_global * ||W||_F^2

        Gradient:
            dRet/dW = 2*lam_local*(W - W_prev) + 2*lam_global*W

        Args:
            W: Current memory [batch, d_key, d_value]
            W_prev: Previous memory [batch, d_key, d_value]

        Returns:
            grad: Retention penalty gradient [batch, d_key, d_value]
            metrics: Observable metrics
        """
        lam_local = self.lambda_local
        lam_global = self.lambda_global

        # Local retention gradient: pull toward previous state
        local_grad = 2 * lam_local * (W - W_prev)

        # Global retention gradient: regularize magnitude
        global_grad = 2 * lam_global * W

        # Combined gradient
        grad = local_grad + global_grad

        # Compute actual penalty values for logging
        local_penalty = lam_local * (W - W_prev).pow(2).sum(dim=(-2, -1)).mean()
        global_penalty = lam_global * W.pow(2).sum(dim=(-2, -1)).mean()

        metrics = {
            "lambda_local": lam_local.item(),
            "lambda_global": lam_global.item(),
            "local_penalty": local_penalty.item(),
            "global_penalty": global_penalty.item(),
            "total_penalty": (local_penalty + global_penalty).item(),
            "retention_grad_norm": grad.norm(dim=(-2, -1)).mean().item(),
        }

        return grad, metrics

    def forward(
        self,
        W: torch.Tensor,
        W_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute retention penalty gradient.

        Args:
            W: Current memory [batch, d_key, d_value]
            W_prev: Previous memory [batch, d_key, d_value]

        Returns:
            grad: Retention penalty gradient
            metrics: Observable metrics
        """
        return self.compute_penalty_gradient(W, W_prev)


class AdaptiveRetentionGate(RetentionGate):
    """
    Retention gate with input-dependent coefficients.

    Extends basic retention with:
    - Per-token retention coefficients based on input
    - "Surprise" modulation: retain less when input is surprising

    This connects to Titans' surprise metric while staying within
    the Miras retention framework.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        d_model: int,
        init_local: float = 0.5,
        init_global: float = 0.1,
    ):
        super().__init__(
            d_key, d_value, init_local, init_global, learn_coefficients=True
        )

        # Project input to retention modulation
        self.local_mod = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        self.global_mod = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def compute_adaptive_coefficients(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute input-dependent retention coefficients.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            local_coef: Local coefficient [batch, seq_len, 1]
            global_coef: Global coefficient [batch, seq_len, 1]
        """
        # Base coefficients
        base_local = self.lambda_local
        base_global = self.lambda_global

        # Input modulation (0.5 to 1.5 range)
        local_mod = 0.5 + self.local_mod(x)  # [batch, seq, 1]
        global_mod = 0.5 + self.global_mod(x)

        # Modulated coefficients
        local_coef = (base_local * local_mod).clamp(0, 1)
        global_coef = (base_global * global_mod).clamp(0, 1)

        return local_coef, global_coef

    def forward_adaptive(
        self,
        W: torch.Tensor,
        W_prev: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute adaptive retention penalty gradient.

        Args:
            W: Current memory [batch, d_key, d_value]
            W_prev: Previous memory [batch, d_key, d_value]
            x: Input for modulation [batch, seq_len, d_model]

        Returns:
            grad: Retention penalty gradient
            metrics: Observable metrics
        """
        local_coef, global_coef = self.compute_adaptive_coefficients(x)

        # Average over sequence for single coefficient per sample
        lam_local = local_coef.mean(dim=1, keepdim=True).squeeze(-1)  # [batch, 1]
        lam_global = global_coef.mean(dim=1, keepdim=True).squeeze(-1)

        # Reshape for broadcasting
        lam_local = lam_local.unsqueeze(-1)  # [batch, 1, 1]
        lam_global = lam_global.unsqueeze(-1)

        # Compute gradients
        local_grad = 2 * lam_local * (W - W_prev)
        global_grad = 2 * lam_global * W
        grad = local_grad + global_grad

        metrics = {
            "lambda_local_mean": lam_local.mean().item(),
            "lambda_global_mean": lam_global.mean().item(),
            "lambda_local_std": local_coef.std().item(),
            "lambda_global_std": global_coef.std().item(),
            "retention_grad_norm": grad.norm(dim=(-2, -1)).mean().item(),
        }

        return grad, metrics


# =============================================================================
# Attention Module (from attention.py)
# =============================================================================


class GateMode(Enum):
    """
    Gate operation modes for episodic memory training.

    NORMAL: Standard learned gating
    STORAGE: Force high gate values (memory writes)
    RETRIEVAL: Ensure minimum gate values (memory reads)
    """

    NORMAL = "normal"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"


class SlidingWindowAttention(nn.Module):
    """
    Multi-head sliding window attention.

    Each position attends only to positions within a fixed window,
    giving O(n * w) complexity instead of O(n^2).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of attention window (one side)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.d_head)

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _create_window_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create causal sliding window mask.

        Returns:
            mask: [seq_len, seq_len] with -inf for masked positions
        """
        # Start with causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

        # Add window constraint: mask positions beyond window
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            if start > 0:
                mask[i, :start] = float("-inf")

        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sliding window attention.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional additional mask
            return_weights: Whether to return attention weights

        Returns:
            output: Attended output [batch, seq_len, d_model]
            weights: Optional attention weights [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply sliding window mask
        window_mask = self._create_window_mask(seq_len, x.device)
        scores = scores + window_mask.unsqueeze(0).unsqueeze(0)

        # Apply additional mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention
        output = torch.matmul(weights, v)

        # Reshape and project
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        if return_weights:
            return output, weights
        return output, None


class GatingMechanism(nn.Module):
    """
    Gating mechanism to combine memory and attention outputs.

    From Titans (MAG variant):
        g = sigmoid(W_g * [mem_out, attn_out])
        output = g * mem_out + (1 - g) * attn_out

    Enhanced with mode-based gate control for episodic memory training:
    - NORMAL: Standard learned gating with optional floor
    - STORAGE: Force high gate values (memory writes)
    - RETRIEVAL: Ensure minimum gate values (memory reads)

    Args:
        d_model: Model dimension
        gate_floor: Minimum gate value (prevents complete memory bypass)
    """

    def __init__(self, d_model: int, gate_floor: float = 0.0):
        super().__init__()

        # Gate projection: takes concatenated inputs
        self.gate_proj = nn.Linear(2 * d_model, d_model, bias=True)

        # Mode and floor settings
        self._mode = GateMode.NORMAL
        self._gate_floor = gate_floor

        # Mode-specific targets
        self._storage_gate_target = 0.8  # During storage, push gate toward this
        self._retrieval_gate_floor = 0.3  # During retrieval, minimum gate

        # Store last gate values for metrics
        self._last_raw_gate: Optional[torch.Tensor] = None
        self._last_gate: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        # Initialize to favor balanced combination
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def set_mode(self, mode: GateMode) -> None:
        """Set the gate operation mode."""
        self._mode = mode

    def get_mode(self) -> GateMode:
        """Get the current gate operation mode."""
        return self._mode

    def set_gate_floor(self, floor: float) -> None:
        """Set the minimum gate value (0.0 to 1.0)."""
        self._gate_floor = max(0.0, min(1.0, floor))

    def get_gate_floor(self) -> float:
        """Get the current gate floor."""
        return self._gate_floor

    def set_storage_target(self, target: float) -> None:
        """Set the target gate value during storage mode."""
        self._storage_gate_target = max(0.0, min(1.0, target))

    def set_retrieval_floor(self, floor: float) -> None:
        """Set the minimum gate value during retrieval mode."""
        self._retrieval_gate_floor = max(0.0, min(1.0, floor))

    def forward(
        self,
        mem_out: torch.Tensor,
        attn_out: torch.Tensor,
        return_gate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Combine memory and attention outputs via gating.

        Args:
            mem_out: Memory output [batch, seq_len, d_model]
            attn_out: Attention output [batch, seq_len, d_model]
            return_gate: Whether to return gate values

        Returns:
            output: Gated combination [batch, seq_len, d_model]
            gate: Optional gate values [batch, seq_len, d_model]
        """
        # Concatenate inputs
        combined = torch.cat([mem_out, attn_out], dim=-1)

        # Compute raw gate (before mode adjustments)
        raw_gate = torch.sigmoid(self.gate_proj(combined))

        # Store raw gate for metrics
        self._last_raw_gate = raw_gate.detach()

        # Apply mode-specific gate constraints
        if self._mode == GateMode.STORAGE:
            # During storage: push gate toward high value (encourage memory writes)
            gate = torch.max(
                raw_gate, torch.full_like(raw_gate, self._storage_gate_target)
            )
        elif self._mode == GateMode.RETRIEVAL:
            # During retrieval: ensure minimum gate (encourage memory reads)
            gate = torch.max(
                raw_gate, torch.full_like(raw_gate, self._retrieval_gate_floor)
            )
        else:
            # Normal mode: apply gate floor
            if self._gate_floor > 0:
                gate = torch.max(
                    raw_gate, torch.full_like(raw_gate, self._gate_floor)
                )
            else:
                gate = raw_gate

        # Store applied gate for metrics
        self._last_gate = gate.detach()

        # Apply gate: high gate = use memory, low gate = use attention
        output = gate * mem_out + (1 - gate) * attn_out

        if return_gate:
            return output, gate
        return output, None

    def get_last_gates(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the last raw and applied gate values.

        Returns:
            Tuple of (raw_gate, applied_gate)
        """
        return self._last_raw_gate, self._last_gate


class FeedForward(nn.Module):
    """
    Standard feed-forward network with GELU activation.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# =============================================================================
# Memory Module (from memory.py)
# =============================================================================


class TitansMemory(nn.Module):
    """
    Titans-style matrix memory with surprise-based updates.

    Implements the correct update equations:
        S_t = eta_t * S_{t-1} - theta_t * grad_loss(M_{t-1}; k_t, v_t)
        M_t = (1 - alpha_t) * M_{t-1} + S_t

    Where alpha_t, eta_t, theta_t are input-dependent via linear projections.

    Args:
        d_model: Model dimension (input dimension)
        d_key: Key/query dimension for memory
        d_value: Value/output dimension for memory
        init_alpha: Initial value for forgetting gate (default 0.1 = 10% forget)
        init_eta: Initial value for surprise decay (default 0.9)
        init_theta: Initial value for gradient scaling (default 0.1)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_key: int = 512,
        d_value: int = 512,
        init_alpha: float = 0.1,
        init_eta: float = 0.9,
        init_theta: float = 0.1,
        # Numerical stability parameters
        grad_clip: float = 1.0,
        memory_max_norm: float = 100.0,
        surprise_max_norm: float = 100.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value

        # Stability parameters
        self.grad_clip = grad_clip
        self.memory_max_norm = memory_max_norm
        self.surprise_max_norm = surprise_max_norm

        # Projections for key, value, query from input
        self.key_proj = nn.Linear(d_model, d_key, bias=False)
        self.value_proj = nn.Linear(d_model, d_value, bias=False)
        self.query_proj = nn.Linear(d_model, d_key, bias=False)

        # Q-K projection: aligns query space to key space (TNT insight)
        # Critical for retrieval to work - queries must be in same space as keys
        self.qk_proj = nn.Linear(d_key, d_key, bias=False)

        # Input-dependent parameter generators (Titans insight)
        # These produce per-token alpha, eta, theta values
        # Use sigmoid to bound outputs to [0, 1]

        # alpha_t: Forgetting gate - how much to decay memory
        self.alpha_proj = nn.Linear(d_model, 1, bias=True)

        # eta_t: Surprise decay - how much past surprise to retain
        self.eta_proj = nn.Linear(d_model, 1, bias=True)

        # theta_t: Gradient scaling - how strongly current gradient affects surprise
        self.theta_proj = nn.Linear(d_model, 1, bias=True)

        # Initialize projection biases to achieve desired initial values
        # sigmoid(bias) ~ init_value, so bias ~ logit(init_value)
        with torch.no_grad():
            self.alpha_proj.bias.fill_(self._logit(init_alpha))
            self.eta_proj.bias.fill_(self._logit(init_eta))
            self.theta_proj.bias.fill_(self._logit(init_theta))

        # Initialize projections
        self._init_weights()

    def _logit(self, p: float) -> float:
        """Compute logit (inverse sigmoid) for initialization."""
        p = max(min(p, 0.999), 0.001)  # Clamp to avoid inf
        return torch.tensor(p / (1 - p)).log().item()

    def _init_weights(self):
        """Initialize projection weights."""
        # Xavier for key/value/query projections
        for proj in [self.key_proj, self.value_proj, self.query_proj]:
            nn.init.xavier_uniform_(proj.weight)

        # Q-K projection starts as identity (TNT)
        nn.init.eye_(self.qk_proj.weight)

        # Small weights for parameter projections (let bias dominate initially)
        for proj in [self.alpha_proj, self.eta_proj, self.theta_proj]:
            nn.init.normal_(proj.weight, std=0.01)

    def _clip_state_norm(
        self,
        state: torch.Tensor,
        max_norm: float,
    ) -> torch.Tensor:
        """
        Clip state tensor norm to prevent explosion.

        Uses per-batch-element clipping to preserve relative magnitudes
        within each batch sample.

        Args:
            state: Tensor [batch, d_key, d_value]
            max_norm: Maximum allowed Frobenius norm per batch element

        Returns:
            Clipped state tensor
        """
        # Compute per-element norm: [batch]
        norms = state.norm(dim=(-2, -1), keepdim=True)  # [batch, 1, 1]

        # Compute scaling factor (only scale down, never up)
        scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)

        return state * scale

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize memory state M and surprise S.

        Returns:
            M: Memory matrix [batch, d_key, d_value] - initialized to small values
            S: Surprise accumulator [batch, d_key, d_value] - initialized to zeros
        """
        # Initialize memory with small random values (not zeros!)
        # This provides a starting point for gradient flow
        M = torch.randn(batch_size, self.d_key, self.d_value, device=device) * 0.01

        # Surprise starts at zero
        S = torch.zeros(batch_size, self.d_key, self.d_value, device=device)

        return M, S

    def compute_gradient(
        self,
        M: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient of L2 attentional bias: loss(M; k, v) = ||M*k - v||^2

        grad_M loss = 2(M*k - v)*k^T

        Args:
            M: Memory [batch, d_key, d_value]
            k: Keys [batch, seq, d_key]
            v: Values [batch, seq, d_value]

        Returns:
            grad: Gradient [batch, d_key, d_value]
            error: Prediction error [batch, d_value, seq]
        """
        # Prediction: M @ k for each position
        # M: [batch, d_key, d_value], k: [batch, seq, d_key]
        # Transpose M to [batch, d_value, d_key], then M @ k^T gives [batch, d_value, seq]
        k_t = k.transpose(-1, -2)  # [batch, d_key, seq]
        pred = torch.bmm(M.transpose(-1, -2), k_t)  # [batch, d_value, seq]

        # Error: pred - v^T
        v_t = v.transpose(-1, -2)  # [batch, d_value, seq]
        error = pred - v_t  # [batch, d_value, seq]

        # Gradient: 2 * k @ error^T (averaged over sequence)
        # k_t: [batch, d_key, seq], error: [batch, d_value, seq]
        # grad should be [batch, d_key, d_value]
        batch_size, seq_len, _ = k.shape
        grad = 2.0 * torch.bmm(k_t, error.transpose(-1, -2)) / seq_len

        return grad, error

    def update(
        self,
        M: torch.Tensor,
        S: torch.Tensor,
        x: torch.Tensor,
        retention_penalty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Update memory using Titans equations:
            S_t = eta_t * S_{t-1} - theta_t * grad_loss(M_{t-1}; k_t, v_t)
            M_t = (1 - alpha_t) * M_{t-1} + S_t

        Args:
            M: Current memory [batch, d_key, d_value]
            S: Current surprise [batch, d_key, d_value]
            x: Input [batch, seq_len, d_model]
            retention_penalty: Optional additional gradient from retention gates

        Returns:
            M_new: Updated memory
            S_new: Updated surprise
            metrics: Observability metrics
        """
        # Project input to keys and values
        k = self.key_proj(x)  # [batch, seq, d_key]
        v = self.value_proj(x)  # [batch, seq, d_value]

        # Compute input-dependent parameters
        # Average over sequence for chunk-level parameters (TNT insight)
        x_mean = x.mean(dim=1)  # [batch, d_model]

        alpha_t = torch.sigmoid(self.alpha_proj(x_mean))  # [batch, 1] - forgetting
        eta_t = torch.sigmoid(self.eta_proj(x_mean))  # [batch, 1] - surprise decay
        theta_t = torch.sigmoid(self.theta_proj(x_mean))  # [batch, 1] - gradient scale

        # Expand for broadcasting with [batch, d_key, d_value]
        alpha_t = alpha_t.unsqueeze(-1)  # [batch, 1, 1]
        eta_t = eta_t.unsqueeze(-1)  # [batch, 1, 1]
        theta_t = theta_t.unsqueeze(-1)  # [batch, 1, 1]

        # Compute gradient
        grad, error = self.compute_gradient(M, k, v)

        # Add retention penalty if provided (Miras)
        if retention_penalty is not None:
            grad = grad + retention_penalty

        # STABILITY: Clip gradient norm to prevent explosion
        # This is critical for persistent memory states
        grad = self._clip_state_norm(grad, self.grad_clip)

        # Titans update equations:
        # S_t = eta_t * S_{t-1} - theta_t * grad_loss
        S_new = eta_t * S - theta_t * grad

        # STABILITY: Clip surprise norm
        S_new = self._clip_state_norm(S_new, self.surprise_max_norm)

        # M_t = (1 - alpha_t) * M_{t-1} + S_t
        M_new = (1 - alpha_t) * M + S_new

        # STABILITY: Clip memory norm
        M_new = self._clip_state_norm(M_new, self.memory_max_norm)

        # Compute metrics for observability
        metrics = {
            "memory_norm": M_new.norm(dim=(-2, -1)).mean().item(),
            "surprise_norm": S_new.norm(dim=(-2, -1)).mean().item(),
            "grad_norm": grad.norm(dim=(-2, -1)).mean().item(),
            "prediction_error": error.abs().mean().item(),
            "alpha_mean": alpha_t.mean().item(),
            "eta_mean": eta_t.mean().item(),
            "theta_mean": theta_t.mean().item(),
        }

        return M_new, S_new, metrics

    def retrieve(
        self,
        M: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from memory using query with Q-K projection (TNT).

        Args:
            M: Memory matrix [batch, d_key, d_value]
            x: Input [batch, seq_len, d_model]

        Returns:
            output: Retrieved values [batch, seq_len, d_value]
        """
        # Project to query
        q = self.query_proj(x)  # [batch, seq, d_key]

        # Q-K projection - align query to key space (TNT critical insight)
        # This ensures retrieval operates in the same space memory was trained on
        q = self.qk_proj(q)  # [batch, seq, d_key]

        # Retrieve: q @ M -> [batch, seq, d_value]
        output = torch.bmm(q, M)

        return output

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        retention_penalty: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[dict]]:
        """
        Full forward pass: update memory then retrieve.

        Args:
            x: Input [batch, seq_len, d_model]
            state: Optional (M, S) tuple
            retention_penalty: Optional retention gradient
            return_metrics: Whether to return metrics

        Returns:
            output: Retrieved values [batch, seq_len, d_value]
            new_state: Updated (M, S) tuple
            metrics: Optional observability metrics
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, device)

        M, S = state

        # Update memory with new information
        M_new, S_new, metrics = self.update(M, S, x, retention_penalty)

        # Retrieve from updated memory
        output = self.retrieve(M_new, x)

        if return_metrics:
            return output, (M_new, S_new), metrics
        return output, (M_new, S_new), None


# Backward compatibility alias
class MatrixMemory(TitansMemory):
    """
    Alias for TitansMemory for backward compatibility.

    Maps old parameter names to new ones:
        d_key -> d_key (same)
        d_value -> d_value (same)
        momentum_beta -> init_eta (surprise decay)
        init_lr -> init_theta (gradient scaling)
        learn_lr -> always True (input-dependent)
    """

    def __init__(
        self,
        d_key: int = 512,
        d_value: int = 512,
        momentum_beta: float = 0.9,
        learn_lr: bool = True,  # Ignored - always input-dependent now
        init_lr: float = 0.1,
        # Pass through stability parameters
        grad_clip: float = 1.0,
        memory_max_norm: float = 100.0,
        surprise_max_norm: float = 100.0,
    ):
        super().__init__(
            d_model=d_key,  # Assume d_model == d_key for compatibility
            d_key=d_key,
            d_value=d_value,
            init_alpha=0.1,  # Default forgetting rate
            init_eta=momentum_beta,  # Map momentum_beta to surprise decay
            init_theta=init_lr,  # Map init_lr to gradient scaling
            grad_clip=grad_clip,
            memory_max_norm=memory_max_norm,
            surprise_max_norm=surprise_max_norm,
        )


class ChunkedTitansMemory(TitansMemory):
    """
    Titans memory with chunk-based processing for TNT-style training.

    Key TNT insights:
    - Compute gradients relative to chunk start state (not per-token state)
    - Q-K projection aligns retrieval to training space
    - Hierarchical: global (large chunks) + local (small chunks) memory
    """

    def __init__(
        self,
        d_model: int = 512,
        d_key: int = 512,
        d_value: int = 512,
        chunk_size: int = 2048,
        init_alpha: float = 0.1,
        init_eta: float = 0.9,
        init_theta: float = 0.1,
        grad_clip: float = 1.0,
        memory_max_norm: float = 100.0,
        surprise_max_norm: float = 100.0,
    ):
        super().__init__(
            d_model=d_model,
            d_key=d_key,
            d_value=d_value,
            init_alpha=init_alpha,
            init_eta=init_eta,
            init_theta=init_theta,
            grad_clip=grad_clip,
            memory_max_norm=memory_max_norm,
            surprise_max_norm=surprise_max_norm,
        )
        self.chunk_size = chunk_size

    def forward_chunked(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[dict]]:
        """
        Chunked forward pass for TNT-style training.

        Within each chunk, all gradients are computed relative to the chunk's
        starting memory state, enabling parallel gradient accumulation.

        Args:
            x: Input [batch, seq_len, d_model]
            state: Global memory state
            return_metrics: Whether to return metrics

        Returns:
            output: Retrieved values
            state: Updated global state
            metrics: Aggregated metrics
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device

        # Initialize global state
        if state is None:
            state = self.init_state(batch_size, device)

        M_global, S_global = state

        # Split into chunks
        chunks = x.split(self.chunk_size, dim=1)

        outputs = []
        all_metrics = []

        for i, chunk in enumerate(chunks):
            # Process chunk with current global state
            output, (M_new, S_new), chunk_metrics = self.forward(
                chunk,
                state=(M_global, S_global),
                return_metrics=return_metrics,
            )
            outputs.append(output)

            # Update global state for next chunk
            # Detach to prevent gradient flow between chunks (TNT)
            M_global = M_new.detach()
            S_global = S_new.detach()

            if return_metrics and chunk_metrics:
                chunk_metrics["chunk_idx"] = i
                all_metrics.append(chunk_metrics)

        # Concatenate outputs
        output = torch.cat(outputs, dim=1)

        # For gradient computation on the final state, re-run the last chunk
        # without detaching its output. This allows gradients to flow through
        # the last chunk's update, but not back to previous chunks (TNT approach).
        if len(chunks) > 0:
            last_chunk = chunks[-1]
            # Use second-to-last state (or initial if only one chunk)
            if len(chunks) > 1:
                # Recompute with gradient flow
                _, (M_global, S_global), _ = self.forward(
                    last_chunk,
                    state=(M_global, S_global),
                    return_metrics=False,
                )

        # Aggregate metrics
        if return_metrics and all_metrics:
            agg_metrics = {
                "n_chunks": len(chunks),
                "final_memory_norm": M_global.norm(dim=(-2, -1)).mean().item(),
                "avg_alpha": sum(m["alpha_mean"] for m in all_metrics) / len(all_metrics),
                "avg_eta": sum(m["eta_mean"] for m in all_metrics) / len(all_metrics),
                "avg_theta": sum(m["theta_mean"] for m in all_metrics) / len(all_metrics),
            }
            return output, (M_global, S_global), agg_metrics

        return output, (M_global, S_global), None


# Keep old name for imports
ChunkedMatrixMemory = ChunkedTitansMemory


# =============================================================================
# Atlas Model (from atlas.py)
# =============================================================================


@dataclass
class AtlasConfig:
    """Configuration for Atlas model."""

    # Model dimensions
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 4
    d_ff: int = 2048
    vocab_size: int = 32000  # LLaMA/Mistral tokenizer vocab size
    max_seq_len: int = 4096

    # Memory configuration
    d_key: int = 512
    d_value: int = 512
    momentum_beta: float = 0.9
    memory_lr_init: float = 0.1
    learn_memory_lr: bool = True

    # Retention configuration
    retention_local_init: float = 0.5
    retention_global_init: float = 0.1
    adaptive_retention: bool = False

    # Attention configuration
    window_size: int = 512

    # Training configuration
    dropout: float = 0.1
    chunk_size: int = 2048  # For TNT training

    # Observability
    log_memory_stats: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0
        assert self.d_key == self.d_model, "d_key must equal d_model for now"
        assert self.d_value == self.d_model, "d_value must equal d_model for now"


class AtlasBlock(nn.Module):
    """
    Single Atlas transformer block.

    Structure:
        1. LayerNorm -> Memory -> Retention penalty
        2. LayerNorm -> Sliding Window Attention
        3. Gate(memory_out, attention_out)
        4. Residual + LayerNorm -> FFN -> Residual

    Args:
        config: AtlasConfig
        layer_idx: Layer index for logging
    """

    def __init__(self, config: AtlasConfig, layer_idx: int = 0):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

        # Memory module
        self.memory = MatrixMemory(
            d_key=config.d_key,
            d_value=config.d_value,
            momentum_beta=config.momentum_beta,
            learn_lr=config.learn_memory_lr,
            init_lr=config.memory_lr_init,
        )

        # Retention gate
        if config.adaptive_retention:
            self.retention = AdaptiveRetentionGate(
                d_key=config.d_key,
                d_value=config.d_value,
                d_model=config.d_model,
                init_local=config.retention_local_init,
                init_global=config.retention_global_init,
            )
        else:
            self.retention = RetentionGate(
                d_key=config.d_key,
                d_value=config.d_value,
                init_local=config.retention_local_init,
                init_global=config.retention_global_init,
            )

        # Attention
        self.attention = SlidingWindowAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            window_size=config.window_size,
            dropout=config.dropout,
        )

        # Gating mechanism
        self.gate = GatingMechanism(config.d_model)

        # Feed-forward
        self.ffn = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        # Dropout for residuals
        self.dropout = nn.Dropout(config.dropout)

        # Dropout for memory output (critical for preventing memorization)
        self.memory_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict]]:
        """
        Forward pass through block.

        Args:
            x: Input [batch, seq_len, d_model]
            memory_state: Optional (W, m) tuple
            return_metrics: Whether to return observability metrics

        Returns:
            output: Block output [batch, seq_len, d_model]
            memory_state: Updated (W, m) tuple
            metrics: Optional dict of metrics
        """
        batch_size = x.shape[0]
        device = x.device
        metrics = {} if return_metrics else None

        # Initialize memory state if needed
        if memory_state is None:
            memory_state = self.memory.init_state(batch_size, device)

        W_prev, m_prev = memory_state

        # === Memory path ===
        x_norm = self.norm1(x)

        # Compute retention penalty gradient
        # First do a "lookahead" update to get W_new for retention computation
        W_temp, m_temp, _ = self.memory.update(W_prev, m_prev, x_norm)

        if isinstance(self.retention, AdaptiveRetentionGate):
            retention_grad, retention_metrics = self.retention.forward_adaptive(
                W_temp, W_prev, x_norm
            )
        else:
            retention_grad, retention_metrics = self.retention(W_temp, W_prev)

        # Now do actual memory update with retention
        mem_out, (W_new, m_new), mem_metrics = self.memory(
            x_norm,
            state=(W_prev, m_prev),
            retention_penalty=retention_grad,
            return_metrics=True,
        )

        # Apply dropout to memory output to prevent memorization
        mem_out = self.memory_dropout(mem_out)

        # === Attention path ===
        x_norm2 = self.norm2(x)
        attn_out, attn_weights = self.attention(x_norm2, return_weights=return_metrics)

        # === Combine via gating ===
        combined, gate_values = self.gate(mem_out, attn_out, return_gate=return_metrics)

        # Residual connection
        x = x + self.dropout(combined)

        # === FFN ===
        x_norm3 = self.norm3(x)
        ffn_out = self.ffn(x_norm3)
        x = x + self.dropout(ffn_out)

        # Collect metrics
        if return_metrics:
            metrics = {
                f"layer_{self.layer_idx}": {
                    "memory": mem_metrics,
                    "retention": retention_metrics,
                    "gate_mean": (
                        gate_values.mean().item() if gate_values is not None else None
                    ),
                    "gate_std": (
                        gate_values.std().item() if gate_values is not None else None
                    ),
                }
            }

        return x, (W_new, m_new), metrics


class Atlas(nn.Module):
    """
    Full Atlas language model.

    Architecture:
        Token Embedding -> Positional Encoding -> N x AtlasBlock -> LM Head

    Uses weight tying between embedding and LM head.

    Args:
        config: AtlasConfig
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()

        self.config = config

        # Token embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [AtlasBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM head (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        # Initialize weights
        self._init_weights()

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Optional[Dict]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            memory_states: Optional list of (W, m) per layer
            return_metrics: Whether to return observability metrics

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            memory_states: Updated memory states per layer
            metrics: Optional dict of metrics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize memory states if needed
        if memory_states is None:
            memory_states = [None] * self.config.n_layers

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # Collect metrics
        all_metrics = {} if return_metrics else None
        new_memory_states = []

        # Forward through blocks
        for i, block in enumerate(self.blocks):
            x, new_state, block_metrics = block(
                x,
                memory_state=memory_states[i],
                return_metrics=return_metrics,
            )
            new_memory_states.append(new_state)

            if return_metrics and block_metrics:
                all_metrics.update(block_metrics)

        # Final norm and LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_memory_states, all_metrics

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        memory_states: Optional[List] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, List, Optional[Dict]]:
        """
        Compute cross-entropy loss.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target token IDs [batch, seq_len]
            memory_states: Optional memory states
            return_metrics: Whether to return metrics

        Returns:
            loss: Scalar loss
            memory_states: Updated memory states
            metrics: Optional metrics including loss/perplexity
        """
        logits, memory_states, metrics = self.forward(
            input_ids,
            memory_states=memory_states,
            return_metrics=return_metrics,
        )

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if return_metrics:
            if metrics is None:
                metrics = {}
            metrics["loss"] = loss.item()
            metrics["perplexity"] = torch.exp(loss).item()

        return loss, memory_states, metrics

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)

        Returns:
            generated: Generated token IDs [batch, seq_len + max_new_tokens]
        """
        memory_states = None

        for _ in range(max_new_tokens):
            # Truncate if needed
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len :]

            # Forward
            logits, memory_states, _ = self.forward(
                input_ids, memory_states=memory_states
            )

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def create_atlas_50m() -> Atlas:
    """Create ~50M parameter Atlas model (paper specs)."""
    config = AtlasConfig(
        d_model=512,
        n_layers=8,
        n_heads=4,
        d_ff=2048,
        vocab_size=32000,  # LLaMA/Mistral tokenizer vocab size
        max_seq_len=4096,
        d_key=512,
        d_value=512,
        momentum_beta=0.9,
        memory_lr_init=0.1,
        learn_memory_lr=True,
        retention_local_init=0.5,
        retention_global_init=0.1,
        adaptive_retention=False,
        window_size=512,
        dropout=0.1,
        chunk_size=2048,
    )
    return Atlas(config)


def create_atlas_100m() -> Atlas:
    """Create ~100M parameter Atlas model."""
    config = AtlasConfig(
        d_model=768,
        n_layers=12,
        n_heads=6,
        d_ff=3072,
        vocab_size=32000,  # LLaMA/Mistral tokenizer vocab size
        max_seq_len=4096,
        d_key=768,
        d_value=768,
        momentum_beta=0.9,
        memory_lr_init=0.1,
        learn_memory_lr=True,
        retention_local_init=0.5,
        retention_global_init=0.1,
        adaptive_retention=False,
        window_size=512,
        dropout=0.1,
        chunk_size=2048,
    )
    return Atlas(config)


# Exports
__all__ = [
    # Config
    "AtlasConfig",
    # Main model
    "Atlas",
    "AtlasBlock",
    # Memory
    "TitansMemory",
    "MatrixMemory",
    "ChunkedTitansMemory",
    "ChunkedMatrixMemory",
    # Retention
    "RetentionGate",
    "AdaptiveRetentionGate",
    # Attention
    "GateMode",
    "SlidingWindowAttention",
    "GatingMechanism",
    "FeedForward",
    # Factory functions
    "create_atlas_50m",
    "create_atlas_100m",
]
