import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Bar, Radar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ChartContainer = ({ results }) => {
  if (!results) return null;

  const { traditional_results, graph_results, comparison_metrics } = results;

  // Handle null values with safe defaults
  const traditionalTime = traditional_results?.retrieval_time || 0;
  const graphTime = graph_results?.retrieval_time || 0;
  
  const tradScores = comparison_metrics?.score_distribution?.traditional || { min: 0, mean: 0, max: 0 };
  const graphScores = comparison_metrics?.score_distribution?.graph || { min: 0, mean: 0, max: 0 };

  const timeChartData = {
    labels: ['Traditional RAG', 'Graph RAG'],
    datasets: [{
      label: 'Retrieval Time (seconds)',
      data: [traditionalTime, graphTime],
      backgroundColor: ['#2196F3', '#4CAF50'],
      borderColor: ['#1976D2', '#388E3C'],
      borderWidth: 2
    }]
  };

  const scoreChartData = {
    labels: ['Min Score', 'Mean Score', 'Max Score'],
    datasets: [{
      label: 'Traditional RAG',
      data: [
        tradScores.min,
        tradScores.mean,
        tradScores.max
      ],
      borderColor: '#2196F3',
      backgroundColor: 'rgba(33, 150, 243, 0.2)',
      pointBackgroundColor: '#2196F3'
    }, {
      label: 'Graph RAG',
      data: [
        graphScores.min,
        graphScores.mean,
        graphScores.max
      ],
      borderColor: '#4CAF50',
      backgroundColor: 'rgba(76, 175, 80, 0.2)',
      pointBackgroundColor: '#4CAF50'
    }]
  };

  const timeChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Time (seconds)'
        }
      }
    }
  };

  const scoreChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom'
      }
    },
    scales: {
      r: {
        beginAtZero: true,
        max: 1
      }
    }
  };

  return (
    <div className="chart-container">
      <h2>📊 Performance Visualization</h2>
      <div className="chart-grid">
        <div className="chart-card">
          <div className="chart-title">Retrieval Speed Comparison</div>
          <Bar data={timeChartData} options={timeChartOptions} />
        </div>
        <div className="chart-card">
          <div className="chart-title">Similarity Score Distribution</div>
          <Radar data={scoreChartData} options={scoreChartOptions} />
        </div>
      </div>
    </div>
  );
};

export default ChartContainer; 