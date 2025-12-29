/**
 * Visualization components barrel exports.
 *
 * Charts for displaying conveyance metrics:
 * - D_eff (Effective Dimension)
 * - Beta (Collapse Indicator)
 * - Alignment (Cosine Similarity)
 * - C_pair (Cross-pair Correlation)
 *
 * Real-time WebSocket support is available via the RealTimeMetricChart
 * component or by setting `realTime={true}` on MetricChart.
 */

// Generic metric chart with optional real-time support
export { MetricChart, RealTimeMetricChart, METRIC_CONFIGS } from './MetricChart';
export type {
  MetricChartProps,
  MetricType,
  MetricConfig,
} from './MetricChart';

// Specialized metric charts
export { DEffChart } from './DEffChart';
export type { DEffChartProps } from './DEffChart';

export { BetaChart } from './BetaChart';
export type { BetaChartProps } from './BetaChart';

export { AlignmentChart } from './AlignmentChart';
export type { AlignmentChartProps } from './AlignmentChart';

export { CPairChart } from './CPairChart';
export type { CPairChartProps } from './CPairChart';
