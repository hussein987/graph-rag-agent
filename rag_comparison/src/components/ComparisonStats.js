import React from 'react';

const ComparisonStats = ({ results }) => {
  if (!results) return null;

  const { traditional_results, graph_results, comparison_metrics } = results;

  // Handle null values with safe defaults
  const traditionalTime = traditional_results?.retrieval_time || 0;
  const graphTime = graph_results?.retrieval_time || 0;
  
  const timeWinner = traditionalTime && graphTime 
    ? (traditionalTime < graphTime ? '🔍 Traditional' : '🕸️ Graph')
    : '⏳ Loading...';

  const overlapPercentage = comparison_metrics?.document_overlap?.overlap_percentage?.toFixed(1) || '0.0';
  const graphSuccess = comparison_metrics?.graph_traversal?.success ? '✅ Success' : '❌ Fallback';
  const totalDocuments = (traditional_results?.total_documents || 0) + (graph_results?.total_documents || 0);

  return (
    <div className="comparison-stats">
      <h2>📈 Comparison Summary</h2>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{timeWinner}</div>
          <div className="stat-label">Performance Winner</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{overlapPercentage}%</div>
          <div className="stat-label">Document Overlap</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{graphSuccess}</div>
          <div className="stat-label">Graph Success Rate</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{totalDocuments}</div>
          <div className="stat-label">Total Documents</div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonStats; 