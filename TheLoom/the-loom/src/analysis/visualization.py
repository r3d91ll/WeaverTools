"""Plotly Visualization Generators for Atlas Model Analysis.

This module provides comprehensive Plotly-based visualization tools for
Atlas model interpretability analysis. It generates interactive 3D landscapes,
animated epoch evolution, and high-quality static exports.

VISUALIZATION CAPABILITIES
===========================
1. 3D Concept Landscapes:
   - Interactive scatter plots showing embedding evolution
   - Fixed axis ranges for smooth animation playback
   - Animated epoch-to-epoch transitions

2. Memory Heatmaps:
   - Layer-wise memory statistics visualization
   - Sparsity, rank, and magnitude displays

3. Evolution Plots:
   - Temporal tracking of metrics across epochs
   - Trend lines and outlier highlighting

4. Export Formats:
   - Interactive HTML (<10MB target)
   - High-resolution PNG (1200x800 @ 300 DPI)

DESIGN DECISIONS
================
- Fixed axis ranges: Ensures smooth animation without jarring rescaling
- Default range [-10, 10]: Appropriate for PCA-transformed embeddings
- PNG export uses scale=2.5 for ~300 DPI at 1200x800 resolution
- HTML includes embedded plotly.js for standalone viewing

Integration: Works with AlignedPCA, ConceptLandscape, and MemoryTracing modules.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Default axis range for 3D visualizations (PCA space)
DEFAULT_AXIS_RANGE: tuple[float, float] = (-10.0, 10.0)

# Default output dimensions (pixels)
DEFAULT_PNG_WIDTH = 1200
DEFAULT_PNG_HEIGHT = 800

# Scale factor for ~300 DPI output
# At 1200x800 pixels with scale=2.5, effective resolution is 3000x2000
# which yields ~300 DPI when printed at 10x6.67 inches
PNG_SCALE_FACTOR = 2.5

# Maximum HTML file size target (bytes)
MAX_HTML_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# Animation defaults
DEFAULT_ANIMATION_DURATION_MS = 500
DEFAULT_FRAME_REDRAW = True

# Color scheme defaults
DEFAULT_COLORSCALE = "viridis"
DEFAULT_MARKER_SIZE = 4
DEFAULT_LINE_WIDTH = 2
DEFAULT_MARKER_OPACITY = 0.8

# WebGL threshold (use WebGL for datasets larger than this)
WEBGL_THRESHOLD_POINTS = 1000


# ============================================================================
# Configuration Data Classes
# ============================================================================


@dataclass
class VisualizationStyle:
    """Style configuration for visualizations.

    Attributes:
        colorscale: Plotly colorscale name for continuous coloring.
        marker_size: Size of scatter markers.
        marker_opacity: Opacity of markers (0.0-1.0).
        line_width: Width of line traces.
        background_color: Scene background color (3D plots).
        font_family: Font family for text elements.
        font_size_title: Font size for titles.
        font_size_axis: Font size for axis labels.
    """

    colorscale: str = DEFAULT_COLORSCALE
    marker_size: int = DEFAULT_MARKER_SIZE
    marker_opacity: float = DEFAULT_MARKER_OPACITY
    line_width: int = DEFAULT_LINE_WIDTH
    background_color: str = "rgb(230, 230, 230)"
    font_family: str = "Arial, sans-serif"
    font_size_title: int = 20
    font_size_axis: int = 12


@dataclass
class AnimationConfig:
    """Animation configuration for temporal visualizations.

    Attributes:
        duration_ms: Duration of each frame transition in milliseconds.
        redraw: Whether to fully redraw on each frame (more reliable but slower).
        mode: Animation mode ('immediate' for smooth transitions).
        easing: Easing function for transitions.
        show_slider: Whether to show the epoch slider control.
        show_play_button: Whether to show play/pause buttons.
    """

    duration_ms: int = DEFAULT_ANIMATION_DURATION_MS
    redraw: bool = DEFAULT_FRAME_REDRAW
    mode: str = "immediate"
    easing: str = "linear"
    show_slider: bool = True
    show_play_button: bool = True


@dataclass
class ExportConfig:
    """Export configuration for visualization outputs.

    Attributes:
        html_path: Path for HTML export (None to skip).
        png_path: Path for PNG export (None to skip).
        png_width: Width of PNG output in pixels.
        png_height: Height of PNG output in pixels.
        png_scale: Scale factor for PNG (2.5 = ~300 DPI).
        include_plotlyjs: Include plotly.js in HTML for standalone viewing.
        optimize_html: Apply size optimizations to HTML output.
    """

    html_path: str | Path | None = None
    png_path: str | Path | None = None
    png_width: int = DEFAULT_PNG_WIDTH
    png_height: int = DEFAULT_PNG_HEIGHT
    png_scale: float = PNG_SCALE_FACTOR
    include_plotlyjs: bool = True
    optimize_html: bool = True


@dataclass
class Axis3DConfig:
    """Configuration for 3D axis ranges.

    Attributes:
        x_range: Range for X axis.
        y_range: Range for Y axis.
        z_range: Range for Z axis.
        x_title: Title for X axis.
        y_title: Title for Y axis.
        z_title: Title for Z axis.
        auto_range: If True, compute range from data (overrides fixed ranges).
        padding_factor: Padding around auto-computed ranges (multiplier).
    """

    x_range: tuple[float, float] = DEFAULT_AXIS_RANGE
    y_range: tuple[float, float] = DEFAULT_AXIS_RANGE
    z_range: tuple[float, float] = DEFAULT_AXIS_RANGE
    x_title: str = "PC1"
    y_title: str = "PC2"
    z_title: str = "PC3"
    auto_range: bool = False
    padding_factor: float = 1.1

    def compute_from_data(
        self,
        data_points: NDArray[np.floating[Any]] | list[NDArray[np.floating[Any]]],
    ) -> "Axis3DConfig":
        """Compute axis ranges from data points.

        Parameters:
            data_points: Single array of shape (n_points, 3) or list of arrays.

        Returns:
            New Axis3DConfig with computed ranges.
        """
        if isinstance(data_points, list):
            all_points = np.concatenate(data_points, axis=0)
        else:
            all_points = data_points

        if all_points.shape[1] < 3:
            raise ValueError(f"Need at least 3D data, got {all_points.shape[1]}D")

        mins = np.min(all_points[:, :3], axis=0)
        maxs = np.max(all_points[:, :3], axis=0)

        # Apply padding
        ranges = maxs - mins
        padded_mins = mins - (self.padding_factor - 1) * ranges / 2
        padded_maxs = maxs + (self.padding_factor - 1) * ranges / 2

        return Axis3DConfig(
            x_range=(float(padded_mins[0]), float(padded_maxs[0])),
            y_range=(float(padded_mins[1]), float(padded_maxs[1])),
            z_range=(float(padded_mins[2]), float(padded_maxs[2])),
            x_title=self.x_title,
            y_title=self.y_title,
            z_title=self.z_title,
            auto_range=False,
            padding_factor=self.padding_factor,
        )


@dataclass
class Landscape3DConfig:
    """Complete configuration for 3D landscape visualizations.

    Attributes:
        title: Title for the visualization.
        axis_config: 3D axis configuration.
        style: Visual style configuration.
        animation: Animation configuration.
        export: Export configuration.
        camera_eye: Camera position (x, y, z).
        show_colorbar: Whether to show colorbar.
        hover_template: Custom hover template.
    """

    title: str = "3D Concept Landscape"
    axis_config: Axis3DConfig = field(default_factory=Axis3DConfig)
    style: VisualizationStyle = field(default_factory=VisualizationStyle)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    camera_eye: tuple[float, float, float] = (1.5, 1.5, 1.5)
    show_colorbar: bool = True
    hover_template: str | None = None


# ============================================================================
# Export Result
# ============================================================================


@dataclass
class ExportResult:
    """Results from visualization export.

    Attributes:
        html_path: Path where HTML was saved (or None if not exported).
        png_path: Path where PNG was saved (or None if not exported).
        html_size_bytes: Size of HTML file in bytes.
        png_size_bytes: Size of PNG file in bytes.
        warnings: Any warnings generated during export.
    """

    html_path: Path | None = None
    png_path: Path | None = None
    html_size_bytes: int = 0
    png_size_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def html_size_mb(self) -> float:
        """HTML file size in megabytes."""
        return self.html_size_bytes / (1024 * 1024)

    @property
    def png_size_mb(self) -> float:
        """PNG file size in megabytes."""
        return self.png_size_bytes / (1024 * 1024)

    @property
    def html_within_limit(self) -> bool:
        """Check if HTML is within the 10MB target."""
        return self.html_size_bytes <= MAX_HTML_SIZE_BYTES


# ============================================================================
# Core 3D Scatter Functions
# ============================================================================


def create_scatter_3d(
    points: NDArray[np.floating[Any]],
    labels: Sequence[str] | None = None,
    colors: NDArray[np.floating[Any]] | Sequence[float] | None = None,
    name: str = "data",
    config: Landscape3DConfig | None = None,
) -> Any:
    """Create a 3D scatter plot.

    Parameters:
        points: Array of shape (n_points, 3) with x, y, z coordinates.
        labels: Text labels for each point (for hover).
        colors: Numeric values for color mapping (length n_points).
        name: Trace name for legend.
        config: Visualization configuration.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    if config is None:
        config = Landscape3DConfig()

    n_points = points.shape[0]

    # Validate input dimensions
    if points.shape[1] < 3:
        raise ValueError(f"Need 3D points, got shape {points.shape}")

    # Use WebGL for large datasets
    use_webgl = n_points > WEBGL_THRESHOLD_POINTS

    # Default colors to point indices
    if colors is None:
        colors = np.arange(n_points)

    # Default labels
    if labels is None:
        labels = [f"Point {i}" for i in range(n_points)]

    # Compute axis ranges if auto_range is True
    axis_config = config.axis_config
    if axis_config.auto_range:
        axis_config = axis_config.compute_from_data(points)

    # Build hover template
    hover_template = config.hover_template
    if hover_template is None:
        hover_template = (
            "%{text}<br>"
            f"{axis_config.x_title}: " + "%{x:.3f}<br>"
            f"{axis_config.y_title}: " + "%{y:.3f}<br>"
            f"{axis_config.z_title}: " + "%{z:.3f}"
            "<extra></extra>"
        )

    # Create trace
    ScatterClass = go.Scatter3d  # Could use scattergl for 2D WebGL  # noqa: N806
    trace = ScatterClass(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        marker=dict(
            size=config.style.marker_size,
            color=colors,
            colorscale=config.style.colorscale,
            opacity=config.style.marker_opacity,
            showscale=config.show_colorbar,
        ),
        text=list(labels),
        hovertemplate=hover_template,
    )

    # Create figure
    fig = go.Figure(data=[trace])

    # Apply layout
    fig.update_layout(
        title=dict(
            text=config.title,
            x=0.5,
            font=dict(
                size=config.style.font_size_title,
                family=config.style.font_family,
            ),
        ),
        scene=dict(
            xaxis=dict(
                range=list(axis_config.x_range),
                title=axis_config.x_title,
                showbackground=True,
                backgroundcolor=config.style.background_color,
            ),
            yaxis=dict(
                range=list(axis_config.y_range),
                title=axis_config.y_title,
                showbackground=True,
                backgroundcolor=config.style.background_color,
            ),
            zaxis=dict(
                range=list(axis_config.z_range),
                title=axis_config.z_title,
                showbackground=True,
                backgroundcolor=config.style.background_color,
            ),
            camera=dict(
                eye=dict(
                    x=config.camera_eye[0],
                    y=config.camera_eye[1],
                    z=config.camera_eye[2],
                ),
            ),
        ),
    )

    return fig


def create_animated_scatter_3d(
    frames_data: list[dict[str, Any]],
    config: Landscape3DConfig | None = None,
) -> Any:
    """Create an animated 3D scatter plot with epoch transitions.

    Parameters:
        frames_data: List of dicts, each containing:
            - 'points': Array of shape (n_points, 3)
            - 'name': Frame name (e.g., "Epoch 50")
            - 'labels': Optional text labels
            - 'colors': Optional color values
            - 'annotation': Optional annotation text
        config: Visualization configuration.

    Returns:
        Plotly Figure object with animation controls.

    Example:
        frames_data = [
            {'points': epoch_0_embeddings[:, :3], 'name': 'Epoch 0'},
            {'points': epoch_50_embeddings[:, :3], 'name': 'Epoch 50'},
            {'points': epoch_100_embeddings[:, :3], 'name': 'Epoch 100'},
        ]
        fig = create_animated_scatter_3d(frames_data)
    """
    import plotly.graph_objects as go

    if not frames_data:
        raise ValueError("frames_data cannot be empty")

    if config is None:
        config = Landscape3DConfig()

    # Compute unified axis ranges from all frames if auto_range
    axis_config = config.axis_config
    if axis_config.auto_range:
        all_points = [frame['points'] for frame in frames_data]
        axis_config = axis_config.compute_from_data(all_points)

    # Build frames
    frames = []
    frame_names = []

    for frame_info in frames_data:
        points = frame_info['points']
        name = frame_info.get('name', f"Frame {len(frames)}")
        labels = frame_info.get('labels', [f"Point {i}" for i in range(len(points))])
        colors = frame_info.get('colors', np.arange(len(points)))
        annotation = frame_info.get('annotation', name)

        hover_template = config.hover_template
        if hover_template is None:
            hover_template = (
                "%{text}<br>"
                f"{axis_config.x_title}: " + "%{x:.3f}<br>"
                f"{axis_config.y_title}: " + "%{y:.3f}<br>"
                f"{axis_config.z_title}: " + "%{z:.3f}"
                "<extra></extra>"
            )

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=config.style.marker_size,
                        color=colors,
                        colorscale=config.style.colorscale,
                        opacity=config.style.marker_opacity,
                    ),
                    text=list(labels),
                    hovertemplate=hover_template,
                )
            ],
            name=name,
            layout=go.Layout(
                annotations=[
                    dict(
                        text=annotation,
                        x=0.5,
                        y=1.05,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14),
                    )
                ]
            ),
        )
        frames.append(frame)
        frame_names.append(name)

    # Initial data from first frame
    initial_points = frames_data[0]['points']
    initial_labels = frames_data[0].get(
        'labels', [f"Point {i}" for i in range(len(initial_points))]
    )
    initial_colors = frames_data[0].get('colors', np.arange(len(initial_points)))

    # Build figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=initial_points[:, 0],
                y=initial_points[:, 1],
                z=initial_points[:, 2],
                mode="markers",
                marker=dict(
                    size=config.style.marker_size,
                    color=initial_colors,
                    colorscale=config.style.colorscale,
                    opacity=config.style.marker_opacity,
                ),
                text=list(initial_labels),
            )
        ],
        frames=frames,
    )

    # Build animation controls
    updatemenus = []
    if config.animation.show_play_button:
        updatemenus.append(
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=config.animation.duration_ms,
                                    redraw=config.animation.redraw,
                                ),
                                fromcurrent=True,
                                mode=config.animation.mode,
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        )

    # Build slider
    sliders = []
    if config.animation.show_slider:
        slider_steps = [
            dict(
                args=[
                    [name],
                    dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                    ),
                ],
                label=name,
                method="animate",
            )
            for name in frame_names
        ]

        sliders.append(
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    prefix="",
                    visible=True,
                    xanchor="center",
                ),
                pad=dict(t=50, b=10),
                len=0.9,
                x=0.1,
                y=0,
                steps=slider_steps,
            )
        )

    # Apply layout
    fig.update_layout(
        title=dict(
            text=config.title,
            x=0.5,
            font=dict(
                size=config.style.font_size_title,
                family=config.style.font_family,
            ),
        ),
        scene=dict(
            xaxis=dict(
                range=list(axis_config.x_range),
                title=axis_config.x_title,
                showbackground=True,
                backgroundcolor=config.style.background_color,
            ),
            yaxis=dict(
                range=list(axis_config.y_range),
                title=axis_config.y_title,
                showbackground=True,
                backgroundcolor=config.style.background_color,
            ),
            zaxis=dict(
                range=list(axis_config.z_range),
                title=axis_config.z_title,
                showbackground=True,
                backgroundcolor=config.style.background_color,
            ),
            camera=dict(
                eye=dict(
                    x=config.camera_eye[0],
                    y=config.camera_eye[1],
                    z=config.camera_eye[2],
                ),
            ),
        ),
        updatemenus=updatemenus,
        sliders=sliders,
    )

    return fig


# ============================================================================
# Trajectory Visualization
# ============================================================================


def create_trajectory_plot_3d(
    trajectories: list[NDArray[np.floating[Any]]],
    labels: Sequence[str] | None = None,
    show_markers: bool = True,
    show_lines: bool = True,
    config: Landscape3DConfig | None = None,
) -> Any:
    """Create a 3D plot showing trajectories through space.

    Parameters:
        trajectories: List of arrays, each shape (n_steps, 3).
        labels: Optional labels for each trajectory.
        show_markers: Whether to show point markers.
        show_lines: Whether to show connecting lines.
        config: Visualization configuration.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    if config is None:
        config = Landscape3DConfig()

    if labels is None:
        labels = [f"Trajectory {i}" for i in range(len(trajectories))]

    # Compute axis ranges if needed
    axis_config = config.axis_config
    if axis_config.auto_range:
        axis_config = axis_config.compute_from_data(trajectories)

    # Build mode string
    mode_parts = []
    if show_markers:
        mode_parts.append("markers")
    if show_lines:
        mode_parts.append("lines")
    mode = "+".join(mode_parts) if mode_parts else "markers"

    fig = go.Figure()

    # Generate colors for trajectories
    n_traj = len(trajectories)
    import plotly.colors as pc
    colors = pc.qualitative.Plotly[:n_traj] if n_traj <= 10 else pc.sample_colorscale(
        'viridis', [i / (n_traj - 1) for i in range(n_traj)]
    )

    for i, (traj, label) in enumerate(zip(trajectories, labels, strict=False)):
        color = colors[i] if i < len(colors) else colors[0]

        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode=mode,
                name=label,
                marker=dict(
                    size=config.style.marker_size,
                    color=color,
                    opacity=config.style.marker_opacity,
                ),
                line=dict(
                    color=color,
                    width=config.style.line_width,
                ),
            )
        )

    # Apply layout
    fig.update_layout(
        title=dict(
            text=config.title,
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(range=list(axis_config.x_range), title=axis_config.x_title),
            yaxis=dict(range=list(axis_config.y_range), title=axis_config.y_title),
            zaxis=dict(range=list(axis_config.z_range), title=axis_config.z_title),
            camera=dict(eye=dict(x=config.camera_eye[0], y=config.camera_eye[1], z=config.camera_eye[2])),
        ),
        showlegend=True,
    )

    return fig


# ============================================================================
# Heatmap Visualizations
# ============================================================================


def create_heatmap(
    data: NDArray[np.floating[Any]],
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    title: str = "Heatmap",
    colorscale: str = "viridis",
    show_values: bool = False,
    value_format: str = ".2f",
) -> Any:
    """Create a 2D heatmap visualization.

    Parameters:
        data: 2D array of values to display.
        x_labels: Labels for x-axis (columns).
        y_labels: Labels for y-axis (rows).
        title: Title for the visualization.
        colorscale: Plotly colorscale name.
        show_values: Whether to show values on cells.
        value_format: Format string for displayed values.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    if x_labels is None:
        x_labels = [str(i) for i in range(data.shape[1])]
    if y_labels is None:
        y_labels = [str(i) for i in range(data.shape[0])]

    # Build text annotation if requested
    text = None
    if show_values:
        text = [[f"{val:{value_format}}" for val in row] for row in data]

    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            x=list(x_labels),
            y=list(y_labels),
            colorscale=colorscale,
            text=text,
            texttemplate="%{text}" if show_values else None,
            hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="", tickangle=45),
        yaxis=dict(title=""),
    )

    return fig


def create_multi_heatmap_grid(
    heatmaps: list[dict[str, Any]],
    n_cols: int = 2,
    main_title: str = "Heatmap Grid",
) -> Any:
    """Create a grid of heatmaps.

    Parameters:
        heatmaps: List of dicts, each containing:
            - 'data': 2D array
            - 'title': Subplot title
            - 'colorscale': Optional colorscale
        n_cols: Number of columns in grid.
        main_title: Main figure title.

    Returns:
        Plotly Figure object.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    n_heatmaps = len(heatmaps)
    n_rows = (n_heatmaps + n_cols - 1) // n_cols

    subplot_titles = [h.get('title', f'Plot {i}') for i, h in enumerate(heatmaps)]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    for i, heatmap in enumerate(heatmaps):
        row = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_trace(
            go.Heatmap(
                z=heatmap['data'],
                colorscale=heatmap.get('colorscale', 'viridis'),
                showscale=(col == n_cols),  # Only show colorbar on rightmost
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=dict(text=main_title, x=0.5),
        height=300 * n_rows,
    )

    return fig


# ============================================================================
# Line Plot Visualizations
# ============================================================================


def create_line_plot(
    x_data: Sequence[float] | NDArray[np.floating[Any]],
    y_data_series: dict[str, Sequence[float] | NDArray[np.floating[Any]]],
    title: str = "Line Plot",
    x_title: str = "X",
    y_title: str = "Y",
    show_markers: bool = True,
    y_axis_type: str = "linear",
) -> Any:
    """Create a line plot with multiple series.

    Parameters:
        x_data: X-axis values.
        y_data_series: Dict mapping series name to y-values.
        title: Plot title.
        x_title: X-axis label.
        y_title: Y-axis label.
        show_markers: Whether to show point markers.
        y_axis_type: 'linear' or 'log'.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    mode = "lines+markers" if show_markers else "lines"

    fig = go.Figure()

    for name, y_data in y_data_series.items():
        fig.add_trace(
            go.Scatter(
                x=list(x_data),
                y=list(y_data),
                mode=mode,
                name=name,
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title, type=y_axis_type),
        showlegend=True,
        legend=dict(x=1.02, y=1),
        hovermode="x unified",
    )

    return fig


def create_multi_line_subplot(
    subplots_data: list[dict[str, Any]],
    main_title: str = "Multi-Line Plots",
    shared_x: bool = True,
) -> Any:
    """Create multiple line plots as subplots.

    Parameters:
        subplots_data: List of dicts, each containing:
            - 'x_data': X-axis values
            - 'y_series': Dict mapping name to y-values
            - 'title': Subplot title
            - 'y_title': Y-axis label
        main_title: Main figure title.
        shared_x: Whether to share x-axis across subplots.

    Returns:
        Plotly Figure object.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    n_subplots = len(subplots_data)

    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        subplot_titles=[s.get('title', '') for s in subplots_data],
        shared_xaxes=shared_x,
        vertical_spacing=0.1,
    )

    for i, subplot in enumerate(subplots_data):
        row = i + 1
        x_data = subplot.get('x_data', [])
        y_series = subplot.get('y_series', {})
        y_title = subplot.get('y_title', '')

        for name, y_data in y_series.items():
            fig.add_trace(
                go.Scatter(
                    x=list(x_data),
                    y=list(y_data),
                    mode="lines+markers",
                    name=name,
                    showlegend=(i == 0),  # Only show legend once
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text=y_title, row=row, col=1)

    if subplots_data:
        fig.update_xaxes(title_text="Epoch", row=n_subplots, col=1)

    fig.update_layout(
        title=dict(text=main_title, x=0.5),
        height=250 * n_subplots,
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    return fig


# ============================================================================
# Surface Plots
# ============================================================================


def create_surface_3d(
    z_data: NDArray[np.floating[Any]],
    x_data: NDArray[np.floating[Any]] | None = None,
    y_data: NDArray[np.floating[Any]] | None = None,
    title: str = "3D Surface",
    colorscale: str = "viridis",
    config: Landscape3DConfig | None = None,
) -> Any:
    """Create a 3D surface plot.

    Parameters:
        z_data: 2D array of Z values.
        x_data: Optional 1D array of X coordinates.
        y_data: Optional 1D array of Y coordinates.
        title: Plot title.
        colorscale: Plotly colorscale name.
        config: Visualization configuration.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    if config is None:
        config = Landscape3DConfig(title=title)

    if x_data is None:
        x_data = np.arange(z_data.shape[1])
    if y_data is None:
        y_data = np.arange(z_data.shape[0])

    fig = go.Figure(
        data=go.Surface(
            z=z_data,
            x=x_data,
            y=y_data,
            colorscale=colorscale,
        )
    )

    fig.update_layout(
        title=dict(text=config.title, x=0.5),
        scene=dict(
            xaxis=dict(title=config.axis_config.x_title),
            yaxis=dict(title=config.axis_config.y_title),
            zaxis=dict(title=config.axis_config.z_title),
            camera=dict(
                eye=dict(
                    x=config.camera_eye[0],
                    y=config.camera_eye[1],
                    z=config.camera_eye[2],
                )
            ),
        ),
    )

    return fig


# ============================================================================
# Export Functions
# ============================================================================


def export_figure(
    fig: Any,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export a Plotly figure to HTML and/or PNG.

    Parameters:
        fig: Plotly Figure object.
        config: Export configuration.

    Returns:
        ExportResult with file paths and sizes.
    """
    if config is None:
        config = ExportConfig()

    result = ExportResult()

    # Export HTML
    if config.html_path is not None:
        html_path = Path(config.html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(
            str(html_path),
            include_plotlyjs=config.include_plotlyjs,
        )

        result.html_path = html_path
        result.html_size_bytes = html_path.stat().st_size

        logger.info(
            f"Exported HTML: {html_path} ({result.html_size_mb:.2f} MB)"
        )

        if not result.html_within_limit:
            result.warnings.append(
                f"HTML size ({result.html_size_mb:.2f} MB) exceeds 10MB target"
            )

    # Export PNG
    if config.png_path is not None:
        png_path = Path(config.png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fig.write_image(
                str(png_path),
                width=config.png_width,
                height=config.png_height,
                scale=config.png_scale,
            )

            result.png_path = png_path
            result.png_size_bytes = png_path.stat().st_size

            # Compute effective DPI
            # scale * width gives pixel width
            # At 1200 * 2.5 = 3000 pixels for a 10-inch print = 300 DPI
            effective_dpi = int(config.png_scale * 72)  # Approximate

            logger.info(
                f"Exported PNG: {png_path} ({result.png_size_mb:.2f} MB, "
                f"{config.png_width}x{config.png_height} @{effective_dpi} DPI)"
            )
        except Exception as e:
            logger.warning(f"PNG export failed: {e}")
            result.warnings.append(f"PNG export failed: {e}")

    return result


def export_for_publication(
    fig: Any,
    base_path: str | Path,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """Export figure in multiple formats suitable for publication.

    Parameters:
        fig: Plotly Figure object.
        base_path: Base path without extension.
        formats: List of formats to export ('html', 'png', 'pdf', 'svg').

    Returns:
        Dict mapping format to output path.
    """
    if formats is None:
        formats = ['html', 'png']

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    for fmt in formats:
        output_path = base_path.with_suffix(f'.{fmt}')

        try:
            if fmt == 'html':
                fig.write_html(str(output_path))
            elif fmt == 'png':
                fig.write_image(
                    str(output_path),
                    width=DEFAULT_PNG_WIDTH,
                    height=DEFAULT_PNG_HEIGHT,
                    scale=PNG_SCALE_FACTOR,
                )
            elif fmt == 'pdf':
                fig.write_image(str(output_path), format='pdf')
            elif fmt == 'svg':
                fig.write_image(str(output_path), format='svg')
            else:
                logger.warning(f"Unknown format: {fmt}")
                continue

            results[fmt] = output_path
            logger.info(f"Exported {fmt.upper()}: {output_path}")

        except Exception as e:
            logger.warning(f"Failed to export {fmt}: {e}")

    return results


# ============================================================================
# High-Level Visualization Functions
# ============================================================================


def visualize_epoch_evolution(
    embeddings_by_epoch: dict[int, NDArray[np.floating[Any]]],
    sample_labels: list[str] | None = None,
    title: str = "Concept Landscape Evolution",
    output_html: str | Path | None = None,
    output_png: str | Path | None = None,
) -> tuple[Any, ExportResult]:
    """Create animated visualization of embedding evolution across epochs.

    This is the main high-level function for concept landscape visualization.

    Parameters:
        embeddings_by_epoch: Dict mapping epoch number to embeddings (n_samples, n_dims).
        sample_labels: Optional text labels for each sample.
        title: Visualization title.
        output_html: Path for HTML export.
        output_png: Path for PNG export (first frame).

    Returns:
        Tuple of (Plotly Figure, ExportResult).
    """
    epochs = sorted(embeddings_by_epoch.keys())
    n_samples = embeddings_by_epoch[epochs[0]].shape[0]

    if sample_labels is None:
        sample_labels = [f"Sample {i}" for i in range(n_samples)]

    # Prepare frames data
    frames_data = []
    for epoch in epochs:
        emb = embeddings_by_epoch[epoch]

        # Ensure 3D
        if emb.shape[1] < 3:
            # Pad with zeros
            emb = np.pad(emb, ((0, 0), (0, 3 - emb.shape[1])))

        frames_data.append({
            'points': emb[:, :3],
            'name': f"Epoch {epoch}",
            'labels': sample_labels,
            'colors': np.arange(n_samples),
            'annotation': f"Epoch {epoch}",
        })

    # Configure visualization
    config = Landscape3DConfig(
        title=title,
        axis_config=Axis3DConfig(auto_range=True),
        export=ExportConfig(
            html_path=output_html,
            png_path=output_png,
        ),
    )

    # Create figure
    fig = create_animated_scatter_3d(frames_data, config)

    # Export
    export_result = export_figure(fig, config.export)

    return fig, export_result


def visualize_memory_statistics(
    layer_stats: list[dict[str, Any]],
    epoch: int = 0,
    output_html: str | Path | None = None,
    output_png: str | Path | None = None,
) -> tuple[Any, ExportResult]:
    """Create visualization of memory statistics across layers.

    Parameters:
        layer_stats: List of dicts with layer statistics, each containing:
            - 'layer_idx': Layer index
            - 'm_sparsity': M matrix sparsity
            - 's_sparsity': S matrix sparsity
            - 'm_effective_rank': M matrix rank
            - 's_effective_rank': S matrix rank
        epoch: Epoch number for title.
        output_html: Path for HTML export.
        output_png: Path for PNG export.

    Returns:
        Tuple of (Plotly Figure, ExportResult).
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    layers = [s['layer_idx'] for s in layer_stats]
    m_sparsity = [s.get('m_sparsity', 0) for s in layer_stats]
    s_sparsity = [s.get('s_sparsity', 0) for s in layer_stats]
    m_rank = [s.get('m_effective_rank', 0) for s in layer_stats]
    s_rank = [s.get('s_effective_rank', 0) for s in layer_stats]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "M Matrix Sparsity",
            "S Matrix Sparsity",
            "M Matrix Effective Rank",
            "S Matrix Effective Rank",
        ),
    )

    fig.add_trace(
        go.Bar(x=layers, y=m_sparsity, marker_color="blue", name="M Sparsity"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=layers, y=s_sparsity, marker_color="green", name="S Sparsity"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=layers, y=m_rank, marker_color="purple", name="M Rank"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=layers, y=s_rank, marker_color="orange", name="S Rank"),
        row=2, col=2,
    )

    fig.update_layout(
        title=dict(text=f"Memory Statistics - Epoch {epoch}", x=0.5),
        height=600,
        showlegend=False,
    )

    export_config = ExportConfig(
        html_path=output_html,
        png_path=output_png,
    )
    export_result = export_figure(fig, export_config)

    return fig, export_result


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Command-line interface for visualization testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plotly Visualization Generators for Atlas Analysis"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test with synthetic data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (HTML)",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=None,
        help="Output path for PNG visualization",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["scatter", "animated", "trajectory", "heatmap", "line"],
        default="animated",
        help="Type of visualization to generate",
    )

    args = parser.parse_args()

    if args.test:
        _run_synthetic_test(args.type, args.output, args.output_png)
    else:
        parser.print_help()
        print("\nUse --test to generate a synthetic visualization")


def _run_synthetic_test(
    viz_type: str,
    output_html: str | None,
    output_png: str | None,
) -> None:
    """Run test with synthetic data."""
    print(f"Generating synthetic {viz_type} visualization...")

    np.random.seed(42)

    if viz_type == "scatter":
        # Simple 3D scatter
        points = np.random.randn(100, 3)
        config = Landscape3DConfig(
            title="Test 3D Scatter",
            axis_config=Axis3DConfig(auto_range=True),
            export=ExportConfig(html_path=output_html, png_path=output_png),
        )
        fig = create_scatter_3d(points, config=config)
        export_figure(fig, config.export)

    elif viz_type == "animated":
        # Animated epoch evolution
        n_samples = 20
        n_features = 128

        # Simulate epoch evolution
        epoch_186 = np.random.randn(n_samples, 3) * 3  # Final state
        embeddings = {
            0: np.random.randn(n_samples, 3) * 5,  # Random initial
            50: epoch_186 + np.random.randn(n_samples, 3) * 2,
            100: epoch_186 + np.random.randn(n_samples, 3) * 1,
            186: epoch_186,
        }

        fig, result = visualize_epoch_evolution(
            embeddings,
            title="Synthetic Epoch Evolution",
            output_html=output_html,
            output_png=output_png,
        )

        print(f"\nExport result:")
        if result.html_path:
            print(f"  HTML: {result.html_path} ({result.html_size_mb:.2f} MB)")
            print(f"  Within 10MB limit: {result.html_within_limit}")
        if result.png_path:
            print(f"  PNG: {result.png_path} ({result.png_size_mb:.2f} MB)")

    elif viz_type == "trajectory":
        # Trajectories through space
        n_traj = 5
        n_steps = 10
        trajectories = [
            np.cumsum(np.random.randn(n_steps, 3), axis=0)
            for _ in range(n_traj)
        ]

        config = Landscape3DConfig(
            title="Test Trajectories",
            axis_config=Axis3DConfig(auto_range=True),
            export=ExportConfig(html_path=output_html, png_path=output_png),
        )
        fig = create_trajectory_plot_3d(trajectories, config=config)
        export_figure(fig, config.export)

    elif viz_type == "heatmap":
        # Multi-heatmap grid
        heatmaps = [
            {'data': np.random.rand(10, 10), 'title': 'Layer 0'},
            {'data': np.random.rand(10, 10), 'title': 'Layer 1'},
            {'data': np.random.rand(10, 10), 'title': 'Layer 2'},
            {'data': np.random.rand(10, 10), 'title': 'Layer 3'},
        ]
        fig = create_multi_heatmap_grid(heatmaps, main_title="Test Heatmaps")

        config = ExportConfig(html_path=output_html, png_path=output_png)
        export_figure(fig, config)

    elif viz_type == "line":
        # Multi-line subplot
        epochs = list(range(0, 200, 10))
        subplots = [
            {
                'x_data': epochs,
                'y_series': {
                    'M Sparsity': np.random.rand(len(epochs)) * 0.5,
                    'S Sparsity': np.random.rand(len(epochs)) * 0.3,
                },
                'title': 'Memory Sparsity',
                'y_title': 'Sparsity',
            },
            {
                'x_data': epochs,
                'y_series': {
                    'M Rank': np.random.randint(10, 50, len(epochs)),
                    'S Rank': np.random.randint(5, 40, len(epochs)),
                },
                'title': 'Effective Rank',
                'y_title': 'Rank',
            },
        ]
        fig = create_multi_line_subplot(subplots, main_title="Test Line Plots")

        config = ExportConfig(html_path=output_html, png_path=output_png)
        export_figure(fig, config)

    print("\nSynthetic test completed successfully!")


if __name__ == "__main__":
    main()
