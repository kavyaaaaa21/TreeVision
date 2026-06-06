import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

export default function ChartsView({ data, speciesColors }) {
  if (!data || !data.features || data.features.length === 0) {
    return (
      <div className="queue-empty">
        <span>No prediction data available.</span>
        <br />
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          Select an image and click "Run prediction" to see telemetry charts.
        </span>
      </div>
    );
  }

  const { features } = data;

  // Chart configuration constants
  const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#8fa891',
          font: { family: "'DM Mono', monospace", size: 11 },
          boxWidth: 12,
          padding: 14,
        },
      },
      tooltip: {
        backgroundColor: '#0c120d',
        borderColor: 'rgba(74, 222, 128, 0.2)',
        borderWidth: 1,
        titleColor: '#e8f0e9',
        bodyColor: '#8fa891',
        titleFont: { family: "'Sora', sans-serif", size: 12, weight: '600' },
        bodyFont: { family: "'DM Mono', monospace", size: 11 },
      },
    },
  };

  const DARK_SCALE = {
    ticks: {
      color: '#5a7060',
      font: { family: "'DM Mono', monospace", size: 10 },
    },
    grid: {
      color: 'rgba(255, 255, 255, 0.03)',
    },
    border: {
      color: 'rgba(255, 255, 255, 0.05)',
    },
  };

  // ── 1. Species Donut Data ──
  const speciesCounts = data.summary?.species_counts || {};
  const donutLabels = Object.keys(speciesCounts);
  const donutValues = Object.values(speciesCounts);
  const donutColors = donutLabels.map((sp) => speciesColors[sp] || '#94A3B8');

  const donutData = {
    labels: donutLabels,
    datasets: [
      {
        data: donutValues,
        backgroundColor: donutColors.map((c) => c + 'cc'),
        borderColor: donutColors,
        borderWidth: 2,
        hoverOffset: 6,
      },
    ],
  };

  const donutOptions = {
    ...CHART_DEFAULTS,
    cutout: '62%',
    plugins: {
      ...CHART_DEFAULTS.plugins,
      legend: {
        ...CHART_DEFAULTS.plugins.legend,
        position: 'right',
      },
    },
  };

  // ── 2. Confidence Histogram Data ──
  const buckets = Array(10).fill(0);
  features.forEach((f) => {
    const idx = Math.min(Math.floor(f.confidence * 10), 9);
    buckets[idx]++;
  });

  const histLabels = buckets.map(
    (_, i) => `${(i * 0.1).toFixed(1)}–${((i + 1) * 0.1).toFixed(1)}`
  );
  const histColors = buckets.map((_, i) => {
    const conf = (i + 0.5) / 10;
    if (conf >= 0.8) return '#4ade80cc'; // Green
    if (conf >= 0.5) return '#fbbf24cc'; // Amber
    return '#f87171cc'; // Red
  });

  const histData = {
    labels: histLabels,
    datasets: [
      {
        label: 'Detections',
        data: buckets,
        backgroundColor: histColors,
        borderColor: histColors.map((c) => c.slice(0, 7)),
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  const histOptions = {
    ...CHART_DEFAULTS,
    scales: {
      x: {
        ...DARK_SCALE,
        title: {
          display: true,
          text: 'Confidence Range',
          color: '#5a7060',
          font: { size: 10, family: "'Sora', sans-serif" },
        },
      },
      y: {
        ...DARK_SCALE,
        beginAtZero: true,
        title: {
          display: true,
          text: 'Tree Count',
          color: '#5a7060',
          font: { size: 10, family: "'Sora', sans-serif" },
        },
      },
    },
    plugins: {
      ...CHART_DEFAULTS.plugins,
      legend: { display: false },
    },
  };

  // ── 3. Mean Crown Area Data ──
  const agg = {};
  features.forEach((f) => {
    if (!agg[f.species]) agg[f.species] = { total: 0, count: 0 };
    agg[f.species].total += f.crown_area_px;
    agg[f.species].count += 1;
  });

  const areaLabels = Object.keys(agg);
  const areaMeans = areaLabels.map((sp) =>
    Math.round(agg[sp].total / agg[sp].count)
  );
  const areaColors = areaLabels.map(
    (sp) => (speciesColors[sp] || '#94A3B8') + 'cc'
  );

  const areaData = {
    labels: areaLabels,
    datasets: [
      {
        label: 'Mean crown area (px²)',
        data: areaMeans,
        backgroundColor: areaColors,
        borderColor: areaColors.map((c) => c.slice(0, 7)),
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  const areaOptions = {
    ...CHART_DEFAULTS,
    indexAxis: 'y',
    scales: {
      x: {
        ...DARK_SCALE,
        beginAtZero: true,
        title: {
          display: true,
          text: 'Pixels (px²)',
          color: '#5a7060',
          font: { size: 10, family: "'Sora', sans-serif" },
        },
      },
      y: DARK_SCALE,
    },
    plugins: {
      ...CHART_DEFAULTS.plugins,
      legend: { display: false },
    },
  };

  return (
    <div className="charts-grid">
      <div className="chart-card">
        <div className="chart-title">Species Distribution</div>
        <div className="chart-canvas-wrap">
          <Doughnut data={donutData} options={donutOptions} />
        </div>
      </div>
      
      <div className="chart-card">
        <div className="chart-title">Confidence Distribution</div>
        <div className="chart-canvas-wrap">
          <Bar data={histData} options={histOptions} />
        </div>
      </div>
      
      <div className="chart-card wide">
        <div className="chart-title">Mean Crown Area by Species (px²)</div>
        <div className="chart-canvas-wrap">
          <Bar data={areaData} options={areaOptions} />
        </div>
      </div>
    </div>
  );
}
