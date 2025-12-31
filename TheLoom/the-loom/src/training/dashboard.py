"""Dash live monitoring dashboard for Atlas training.

This module provides a real-time training dashboard using Dash/Plotly
that displays live training metrics including loss curves, learning rate
schedule, GPU utilization, and memory statistics.

Dashboard runs in a separate process to avoid blocking training (<1% overhead).
Metrics are read from a JSON lines file written by AtlasTrainer.

Usage:
    # Start dashboard server (separate terminal)
    poetry run python -m src.training.dashboard --metrics-file /tmp/metrics.jsonl

    # Or with custom port
    poetry run python -m src.training.dashboard --metrics-file /tmp/metrics.jsonl --port 8050

References:
- AtlasTrainer: TheLoom/the-loom/src/training/atlas_trainer.py
- Metrics format: TrainingMetrics dataclass written as JSON lines
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Dash and Plotly imports
try:
    import dash
    from dash import dcc, html, callback, Input, Output, State
    from dash.exceptions import PreventUpdate
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"Error: Dash not installed. Install with: poetry add dash")
    print(f"Import error: {e}")
    sys.exit(1)

# Add parent paths for imports when running as module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Dashboard update interval in milliseconds (1 Hz = 1000ms)
UPDATE_INTERVAL_MS = 1000

# Maximum number of data points to keep in memory
MAX_DATA_POINTS = 500

# Chart layout configuration
CHART_HEIGHT = 300
CHART_MARGIN = dict(l=60, r=30, t=40, b=40)

# Color palette matching TheLoom branding
COLORS = {
    "background": "#1E1E2E",
    "card_bg": "#2A2A3E",
    "text": "#FFFFFF",
    "text_secondary": "#A0A0B0",
    "primary": "#6366F1",  # Indigo
    "success": "#10B981",  # Green
    "warning": "#F59E0B",  # Amber
    "error": "#EF4444",    # Red
    "line_loss": "#6366F1",
    "line_lr": "#10B981",
    "line_memory": "#F59E0B",
    "line_gpu": "#EF4444",
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DashboardMetrics:
    """Parsed metrics from the training log.

    Mirrors TrainingMetrics from atlas_trainer.py for dashboard display.
    """
    loss: float = 0.0
    perplexity: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    step: int = 0
    tokens_per_second: float = 0.0
    gpu_memory_mb: float = 0.0
    memory_norm: float = 0.0
    timestamp: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DashboardMetrics":
        """Create from dictionary."""
        return cls(
            loss=data.get("loss", 0.0),
            perplexity=data.get("perplexity", 0.0),
            learning_rate=data.get("learning_rate", 0.0),
            epoch=data.get("epoch", 0),
            step=data.get("step", 0),
            tokens_per_second=data.get("tokens_per_second", 0.0),
            gpu_memory_mb=data.get("gpu_memory_mb", 0.0),
            memory_norm=data.get("memory_norm", 0.0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class MetricsBuffer:
    """Thread-safe buffer for streaming metrics.

    Provides caching for connection loss recovery and efficient
    data access for dashboard updates.
    """
    max_size: int = MAX_DATA_POINTS
    _data: deque = field(default_factory=lambda: deque(maxlen=MAX_DATA_POINTS))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_read_position: int = 0

    def add(self, metrics: DashboardMetrics) -> None:
        """Add metrics to the buffer (thread-safe)."""
        with self._lock:
            self._data.append(metrics)

    def get_all(self) -> list[DashboardMetrics]:
        """Get all buffered metrics (thread-safe)."""
        with self._lock:
            return list(self._data)

    def get_latest(self) -> Optional[DashboardMetrics]:
        """Get the most recent metrics (thread-safe)."""
        with self._lock:
            return self._data[-1] if self._data else None

    def clear(self) -> None:
        """Clear the buffer (thread-safe)."""
        with self._lock:
            self._data.clear()
            self._last_read_position = 0

    @property
    def size(self) -> int:
        """Current number of items in buffer."""
        with self._lock:
            return len(self._data)

    @property
    def last_read_position(self) -> int:
        """Last file position read."""
        with self._lock:
            return self._last_read_position

    @last_read_position.setter
    def last_read_position(self, value: int) -> None:
        """Set last file position read."""
        with self._lock:
            self._last_read_position = value


# ============================================================================
# Metrics File Reader
# ============================================================================

class MetricsFileReader:
    """Reads training metrics from a JSON lines file.

    Watches the metrics file for new data and updates the buffer.
    Uses file position tracking for efficient incremental reads.
    """

    def __init__(
        self,
        metrics_file: str | Path,
        buffer: MetricsBuffer,
    ):
        """Initialize the reader.

        Args:
            metrics_file: Path to metrics JSON lines file
            buffer: Buffer to store parsed metrics
        """
        self.metrics_file = Path(metrics_file)
        self.buffer = buffer
        self._file_position = 0
        self._last_mtime = 0.0

    def read_new_metrics(self) -> int:
        """Read new metrics from file since last read.

        Returns:
            Number of new metrics read
        """
        if not self.metrics_file.exists():
            return 0

        # Check if file was modified
        try:
            current_mtime = self.metrics_file.stat().st_mtime
        except OSError:
            return 0

        if current_mtime <= self._last_mtime:
            return 0

        self._last_mtime = current_mtime

        # Read new lines from last position
        count = 0
        try:
            with open(self.metrics_file, "r") as f:
                f.seek(self._file_position)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        metrics = DashboardMetrics.from_dict(data)
                        self.buffer.add(metrics)
                        count += 1
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

                self._file_position = f.tell()
        except OSError as e:
            logger.warning(f"Error reading metrics file: {e}")
            return 0

        return count

    def reset(self) -> None:
        """Reset reader state for file rotation."""
        self._file_position = 0
        self._last_mtime = 0.0
        self.buffer.clear()


# ============================================================================
# Chart Creators
# ============================================================================

def create_loss_chart(
    metrics: list[DashboardMetrics],
) -> go.Figure:
    """Create loss curve chart.

    Args:
        metrics: List of metrics data points

    Returns:
        Plotly figure with loss and perplexity curves
    """
    if not metrics:
        # Empty chart placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for training data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"]),
        )
    else:
        steps = [m.step for m in metrics]
        losses = [m.loss for m in metrics]
        perplexities = [min(m.perplexity, 1000) for m in metrics]  # Clamp for display

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Training Loss", "Perplexity"),
            horizontal_spacing=0.12,
        )

        # Loss trace
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=losses,
                mode="lines",
                name="Loss",
                line=dict(color=COLORS["line_loss"], width=2),
                hovertemplate="Step: %{x}<br>Loss: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )

        # Perplexity trace
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=perplexities,
                mode="lines",
                name="Perplexity",
                line=dict(color=COLORS["warning"], width=2),
                hovertemplate="Step: %{x}<br>PPL: %{y:.2f}<extra></extra>",
            ),
            row=1, col=2,
        )

        fig.update_xaxes(title_text="Step", row=1, col=1)
        fig.update_xaxes(title_text="Step", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Perplexity", row=1, col=2)

    fig.update_layout(
        height=CHART_HEIGHT,
        margin=CHART_MARGIN,
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text"]),
        showlegend=False,
    )

    return fig


def create_lr_chart(
    metrics: list[DashboardMetrics],
) -> go.Figure:
    """Create learning rate schedule chart.

    Args:
        metrics: List of metrics data points

    Returns:
        Plotly figure with LR schedule
    """
    fig = go.Figure()

    if not metrics:
        fig.add_annotation(
            text="Waiting for training data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"]),
        )
    else:
        steps = [m.step for m in metrics]
        lrs = [m.learning_rate for m in metrics]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=lrs,
                mode="lines",
                name="Learning Rate",
                line=dict(color=COLORS["line_lr"], width=2),
                fill="tozeroy",
                fillcolor=f"rgba(16, 185, 129, 0.1)",
                hovertemplate="Step: %{x}<br>LR: %{y:.2e}<extra></extra>",
            )
        )

        fig.update_xaxes(title_text="Step")
        fig.update_yaxes(title_text="Learning Rate", tickformat=".1e")

    fig.update_layout(
        title="Learning Rate Schedule",
        height=CHART_HEIGHT,
        margin=CHART_MARGIN,
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text"]),
        showlegend=False,
    )

    return fig


def create_gpu_chart(
    metrics: list[DashboardMetrics],
) -> go.Figure:
    """Create GPU memory and tokens/s chart.

    Args:
        metrics: List of metrics data points

    Returns:
        Plotly figure with GPU stats
    """
    if not metrics:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for training data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"]),
        )
    else:
        steps = [m.step for m in metrics]
        gpu_mb = [m.gpu_memory_mb for m in metrics]
        tokens_s = [m.tokens_per_second for m in metrics]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("GPU Memory (MB)", "Tokens/Second"),
            horizontal_spacing=0.12,
        )

        # GPU Memory
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=gpu_mb,
                mode="lines",
                name="GPU Memory",
                line=dict(color=COLORS["line_gpu"], width=2),
                fill="tozeroy",
                fillcolor=f"rgba(239, 68, 68, 0.1)",
                hovertemplate="Step: %{x}<br>GPU: %{y:.0f} MB<extra></extra>",
            ),
            row=1, col=1,
        )

        # Tokens/s
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=tokens_s,
                mode="lines",
                name="Tokens/s",
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy",
                fillcolor=f"rgba(99, 102, 241, 0.1)",
                hovertemplate="Step: %{x}<br>Tokens/s: %{y:.0f}<extra></extra>",
            ),
            row=1, col=2,
        )

        fig.update_xaxes(title_text="Step", row=1, col=1)
        fig.update_xaxes(title_text="Step", row=1, col=2)
        fig.update_yaxes(title_text="MB", row=1, col=1)
        fig.update_yaxes(title_text="Tokens/s", row=1, col=2)

    fig.update_layout(
        height=CHART_HEIGHT,
        margin=CHART_MARGIN,
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text"]),
        showlegend=False,
    )

    return fig


def create_memory_chart(
    metrics: list[DashboardMetrics],
) -> go.Figure:
    """Create memory matrix norm chart.

    Args:
        metrics: List of metrics data points

    Returns:
        Plotly figure with memory statistics
    """
    fig = go.Figure()

    if not metrics:
        fig.add_annotation(
            text="Waiting for training data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"]),
        )
    else:
        steps = [m.step for m in metrics]
        memory_norms = [m.memory_norm for m in metrics]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=memory_norms,
                mode="lines",
                name="Memory Norm",
                line=dict(color=COLORS["line_memory"], width=2),
                fill="tozeroy",
                fillcolor=f"rgba(245, 158, 11, 0.1)",
                hovertemplate="Step: %{x}<br>Norm: %{y:.4f}<extra></extra>",
            )
        )

        fig.update_xaxes(title_text="Step")
        fig.update_yaxes(title_text="Average Memory Norm")

    fig.update_layout(
        title="Memory Matrix Statistics",
        height=CHART_HEIGHT,
        margin=CHART_MARGIN,
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text"]),
        showlegend=False,
    )

    return fig


# ============================================================================
# Dashboard Application
# ============================================================================

class TrainingDashboard:
    """Dash application for live training monitoring.

    Provides a real-time web dashboard that displays training metrics
    updated at 1 Hz with minimal overhead.

    Features:
    - Loss and perplexity curves
    - Learning rate schedule visualization
    - GPU memory and throughput monitoring
    - Memory matrix statistics
    - Status indicators for training progress

    Example:
        dashboard = TrainingDashboard(metrics_file="/tmp/metrics.jsonl")
        dashboard.run(port=8050)
    """

    def __init__(
        self,
        metrics_file: str | Path,
        update_interval_ms: int = UPDATE_INTERVAL_MS,
        debug: bool = False,
    ):
        """Initialize the dashboard.

        Args:
            metrics_file: Path to metrics JSON lines file
            update_interval_ms: Update interval in milliseconds (default 1000 for 1 Hz)
            debug: Enable Dash debug mode
        """
        self.metrics_file = Path(metrics_file)
        self.update_interval_ms = update_interval_ms
        self.debug = debug

        # Initialize metrics buffer and reader
        self.buffer = MetricsBuffer()
        self.reader = MetricsFileReader(metrics_file, self.buffer)

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            title="Atlas Training Dashboard",
            update_title=None,  # Disable "Updating..." in title
            suppress_callback_exceptions=True,
        )

        # Build layout
        self.app.layout = self._create_layout()

        # Register callbacks
        self._register_callbacks()

    def _create_layout(self) -> html.Div:
        """Create the dashboard layout.

        Returns:
            Dash HTML layout
        """
        return html.Div(
            style={
                "backgroundColor": COLORS["background"],
                "minHeight": "100vh",
                "padding": "20px",
                "fontFamily": "system-ui, -apple-system, sans-serif",
            },
            children=[
                # Interval component for 1 Hz updates
                dcc.Interval(
                    id="update-interval",
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),

                # Store for metrics data
                dcc.Store(id="metrics-store"),

                # Header
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "20px",
                    },
                    children=[
                        html.H1(
                            "Atlas Training Dashboard",
                            style={
                                "color": COLORS["text"],
                                "margin": "0",
                                "fontSize": "24px",
                                "fontWeight": "600",
                            },
                        ),
                        html.Div(
                            id="status-indicator",
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "10px",
                            },
                        ),
                    ],
                ),

                # Stats cards row
                html.Div(
                    id="stats-cards",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                        "gap": "15px",
                        "marginBottom": "20px",
                    },
                ),

                # Charts grid
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(2, 1fr)",
                        "gap": "15px",
                    },
                    children=[
                        # Loss chart
                        html.Div(
                            style={
                                "backgroundColor": COLORS["card_bg"],
                                "borderRadius": "8px",
                                "padding": "15px",
                            },
                            children=[
                                dcc.Graph(
                                    id="loss-chart",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                ),
                            ],
                        ),

                        # LR chart
                        html.Div(
                            style={
                                "backgroundColor": COLORS["card_bg"],
                                "borderRadius": "8px",
                                "padding": "15px",
                            },
                            children=[
                                dcc.Graph(
                                    id="lr-chart",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                ),
                            ],
                        ),

                        # GPU chart
                        html.Div(
                            style={
                                "backgroundColor": COLORS["card_bg"],
                                "borderRadius": "8px",
                                "padding": "15px",
                            },
                            children=[
                                dcc.Graph(
                                    id="gpu-chart",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                ),
                            ],
                        ),

                        # Memory chart
                        html.Div(
                            style={
                                "backgroundColor": COLORS["card_bg"],
                                "borderRadius": "8px",
                                "padding": "15px",
                            },
                            children=[
                                dcc.Graph(
                                    id="memory-chart",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),

                # Footer
                html.Div(
                    style={
                        "marginTop": "20px",
                        "textAlign": "center",
                        "color": COLORS["text_secondary"],
                        "fontSize": "12px",
                    },
                    children=[
                        html.Span(f"Metrics file: {self.metrics_file}"),
                        html.Span(" | ", style={"margin": "0 10px"}),
                        html.Span(f"Update rate: {1000 / self.update_interval_ms:.1f} Hz"),
                    ],
                ),
            ],
        )

    def _create_stat_card(
        self,
        title: str,
        value: str,
        subtitle: str = "",
        color: str = COLORS["primary"],
    ) -> html.Div:
        """Create a statistics card component.

        Args:
            title: Card title
            value: Main value to display
            subtitle: Optional subtitle
            color: Accent color

        Returns:
            Dash HTML div for the card
        """
        return html.Div(
            style={
                "backgroundColor": COLORS["card_bg"],
                "borderRadius": "8px",
                "padding": "15px",
                "borderLeft": f"4px solid {color}",
            },
            children=[
                html.Div(
                    title,
                    style={
                        "color": COLORS["text_secondary"],
                        "fontSize": "12px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px",
                    },
                ),
                html.Div(
                    value,
                    style={
                        "color": COLORS["text"],
                        "fontSize": "24px",
                        "fontWeight": "600",
                        "marginTop": "5px",
                    },
                ),
                html.Div(
                    subtitle,
                    style={
                        "color": COLORS["text_secondary"],
                        "fontSize": "12px",
                        "marginTop": "5px",
                    },
                ) if subtitle else None,
            ],
        )

    def _register_callbacks(self) -> None:
        """Register Dash callbacks for live updates."""

        @self.app.callback(
            [
                Output("metrics-store", "data"),
                Output("status-indicator", "children"),
                Output("stats-cards", "children"),
                Output("loss-chart", "figure"),
                Output("lr-chart", "figure"),
                Output("gpu-chart", "figure"),
                Output("memory-chart", "figure"),
            ],
            [Input("update-interval", "n_intervals")],
        )
        def update_dashboard(n_intervals: int):
            """Update all dashboard components.

            This callback is triggered every update_interval_ms milliseconds.
            It reads new metrics from the file and updates all visualizations.

            Args:
                n_intervals: Number of update intervals elapsed

            Returns:
                Tuple of updated component values
            """
            # Read new metrics
            new_count = self.reader.read_new_metrics()
            metrics = self.buffer.get_all()
            latest = self.buffer.get_latest()

            # Status indicator
            if latest:
                status_children = [
                    html.Div(
                        style={
                            "width": "10px",
                            "height": "10px",
                            "borderRadius": "50%",
                            "backgroundColor": COLORS["success"],
                            "animation": "pulse 2s infinite",
                        },
                    ),
                    html.Span(
                        f"Epoch {latest.epoch} | Step {latest.step}",
                        style={
                            "color": COLORS["text"],
                            "fontSize": "14px",
                        },
                    ),
                    html.Span(
                        f"Last update: {latest.timestamp[:19] if latest.timestamp else 'N/A'}",
                        style={
                            "color": COLORS["text_secondary"],
                            "fontSize": "12px",
                            "marginLeft": "10px",
                        },
                    ),
                ]
            else:
                status_children = [
                    html.Div(
                        style={
                            "width": "10px",
                            "height": "10px",
                            "borderRadius": "50%",
                            "backgroundColor": COLORS["warning"],
                        },
                    ),
                    html.Span(
                        "Waiting for training data...",
                        style={
                            "color": COLORS["text_secondary"],
                            "fontSize": "14px",
                        },
                    ),
                ]

            # Stats cards
            if latest:
                stats_cards = [
                    self._create_stat_card(
                        "Current Loss",
                        f"{latest.loss:.4f}",
                        f"PPL: {latest.perplexity:.2f}",
                        COLORS["line_loss"],
                    ),
                    self._create_stat_card(
                        "Learning Rate",
                        f"{latest.learning_rate:.2e}",
                        "",
                        COLORS["line_lr"],
                    ),
                    self._create_stat_card(
                        "GPU Memory",
                        f"{latest.gpu_memory_mb:.0f} MB",
                        f"{latest.tokens_per_second:.0f} tokens/s",
                        COLORS["line_gpu"],
                    ),
                    self._create_stat_card(
                        "Memory Norm",
                        f"{latest.memory_norm:.4f}",
                        f"{len(metrics)} data points",
                        COLORS["line_memory"],
                    ),
                ]
            else:
                stats_cards = [
                    self._create_stat_card("Current Loss", "---", "", COLORS["text_secondary"]),
                    self._create_stat_card("Learning Rate", "---", "", COLORS["text_secondary"]),
                    self._create_stat_card("GPU Memory", "---", "", COLORS["text_secondary"]),
                    self._create_stat_card("Memory Norm", "---", "", COLORS["text_secondary"]),
                ]

            # Create charts
            loss_fig = create_loss_chart(metrics)
            lr_fig = create_lr_chart(metrics)
            gpu_fig = create_gpu_chart(metrics)
            memory_fig = create_memory_chart(metrics)

            # Return serialized metrics for store (not used directly but available)
            metrics_data = {"count": len(metrics)}

            return (
                metrics_data,
                status_children,
                stats_cards,
                loss_fig,
                lr_fig,
                gpu_fig,
                memory_fig,
            )

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8050,
    ) -> None:
        """Run the dashboard server.

        Args:
            host: Host address to bind to
            port: Port number
        """
        logger.info(f"Starting Atlas Training Dashboard on http://{host}:{port}")
        logger.info(f"Metrics file: {self.metrics_file}")
        logger.info(f"Update rate: {1000 / self.update_interval_ms:.1f} Hz")

        self.app.run(
            host=host,
            port=port,
            debug=self.debug,
            use_reloader=False,  # Disable reloader to avoid issues with threading
        )


# ============================================================================
# Demo Mode
# ============================================================================

def generate_demo_metrics(output_file: Path, duration_seconds: int = 30) -> None:
    """Generate demo metrics for testing the dashboard.

    Creates a metrics file with synthetic training data that simulates
    a realistic training run.

    Args:
        output_file: Path to write demo metrics
        duration_seconds: How long to generate metrics
    """
    import math
    import random

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating demo metrics to {output_file}...")
    print(f"Dashboard should be accessible at http://localhost:8050")
    print("Press Ctrl+C to stop")

    step = 0
    epoch = 0
    base_loss = 4.0

    try:
        with open(output_file, "w") as f:
            while True:
                # Simulate decreasing loss with noise
                progress = step / 1000.0
                loss = base_loss * math.exp(-progress * 0.5) + random.gauss(0, 0.1)
                loss = max(0.1, loss)

                perplexity = math.exp(min(loss, 10))

                # LR with warmup and decay
                warmup_steps = 100
                if step < warmup_steps:
                    lr = 1e-4 * (step / warmup_steps)
                else:
                    lr = 1e-4 * (0.5 + 0.5 * math.cos(math.pi * (step - warmup_steps) / 1000))

                # GPU memory oscillates slightly
                gpu_mb = 2500 + random.gauss(0, 100)

                # Tokens/s with some variation
                tokens_s = 8000 + random.gauss(0, 500)

                # Memory norm grows then stabilizes
                memory_norm = 1.0 + 0.5 * math.tanh(step / 200) + random.gauss(0, 0.05)

                # Create metrics
                metrics = {
                    "loss": loss,
                    "perplexity": perplexity,
                    "learning_rate": lr,
                    "epoch": epoch,
                    "step": step,
                    "tokens_per_second": tokens_s,
                    "gpu_memory_mb": gpu_mb,
                    "memory_norm": memory_norm,
                    "timestamp": datetime.now().isoformat(),
                }

                f.write(json.dumps(metrics) + "\n")
                f.flush()

                step += 1
                if step % 50 == 0:
                    epoch += 1

                time.sleep(1.0)  # 1 Hz update

    except KeyboardInterrupt:
        print("\nDemo stopped")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """CLI entry point for the training dashboard."""
    parser = argparse.ArgumentParser(
        description="Atlas Training Dashboard - Live monitoring for training runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to metrics JSON lines file from training",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run dashboard on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to",
    )
    parser.add_argument(
        "--update-rate",
        type=float,
        default=1.0,
        help="Update rate in Hz (default 1.0 = 1 update per second)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic data",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Demo mode
    if args.demo:
        demo_file = Path("/tmp/atlas_demo_metrics.jsonl")

        # Start demo data generator in background thread
        import threading
        demo_thread = threading.Thread(
            target=generate_demo_metrics,
            args=(demo_file, 300),
            daemon=True,
        )
        demo_thread.start()

        # Wait for first metrics
        time.sleep(2)

        # Start dashboard
        dashboard = TrainingDashboard(
            metrics_file=demo_file,
            update_interval_ms=int(1000 / args.update_rate),
            debug=args.debug,
        )
        dashboard.run(host=args.host, port=args.port)
        return

    # Normal mode
    if not args.metrics_file:
        parser.error("--metrics-file is required (or use --demo for demo mode)")

    metrics_file = Path(args.metrics_file)

    # Create parent directory if needed
    if not metrics_file.parent.exists():
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Warn if file doesn't exist yet
    if not metrics_file.exists():
        logger.warning(f"Metrics file does not exist yet: {metrics_file}")
        logger.info("Dashboard will wait for training to start writing metrics...")

    # Calculate update interval
    update_interval_ms = int(1000 / args.update_rate)

    # Create and run dashboard
    dashboard = TrainingDashboard(
        metrics_file=metrics_file,
        update_interval_ms=update_interval_ms,
        debug=args.debug,
    )

    dashboard.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
