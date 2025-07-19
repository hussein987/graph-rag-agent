import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const DocumentItem = ({ doc, index }) => {
  const source = doc.metadata.source || 'Unknown';
  const score = doc.metadata.similarity_score;
  const [isExpanded, setIsExpanded] = useState(false);
  
  const toggleExpand = () => setIsExpanded(!isExpanded);
  
  // Check if this is a GraphRAG result
  const isGraphRAG = source.includes('working_graphrag');
  
  // Check if this is an LLM-generated answer
  const isLLMAnswer = doc.metadata.is_generated_answer === true;
  
  const renderContent = (content, isFullContent = false) => {
    if (isFullContent) {
      // For expanded results, render both traditional and GraphRAG as markdown
      return (
        <div className="markdown-content">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      );
    } else {
      // For collapsed results, show plain text preview
      return (
        <>
          {content.substring(0, 200)}
          {content.length > 200 ? '...' : ''}
        </>
      );
    }
  };
  
  const getDocumentIcon = () => {
    if (isLLMAnswer) return '🤖';
    if (isGraphRAG) return '🕸️';
    return '📄';
  };
  
  const getDocumentLabel = () => {
    if (isLLMAnswer) return 'LLM Generated Answer';
    if (isGraphRAG) return source.split('/').pop();
    return source.split('/').pop();
  };
  
  return (
    <div className={`document-item ${isGraphRAG ? 'graphrag-item' : isLLMAnswer ? 'llm-answer-item' : 'traditional-item'}`}>
      <div className="document-meta">
        <span>{getDocumentIcon()} {getDocumentLabel()}</span>
        {score !== undefined && (
          <span className="score-badge">{score.toFixed(3)}</span>
        )}
        <button 
          className="expand-button" 
          onClick={toggleExpand}
          aria-label={isExpanded ? 'Collapse' : 'Expand'}
        >
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>
      <div className={`document-content ${isExpanded ? 'expanded' : 'collapsed'}`}>
        {renderContent(doc.page_content, isExpanded)}
      </div>
      {doc.metadata.search_type && (
        <div className="document-search-type">
          <span className={`search-type-badge ${doc.metadata.search_type}`}>
            {doc.metadata.search_type} search
          </span>
        </div>
      )}
      {doc.metadata.context_text && isExpanded && (
        <div className="document-context">
          <h4>Context Used:</h4>
          <pre className="context-text">{doc.metadata.context_text}</pre>
        </div>
      )}
    </div>
  );
};

const ComparisonContainer = ({ results }) => {
  if (!results) return null;

  const { traditional_results, graph_results, comparison_metrics } = results;

  const renderDocuments = (documents) => {
    if (!documents || documents.length === 0) {
      return <div className="no-documents">No documents available yet...</div>;
    }
    
    return documents.map((doc, index) => (
      <DocumentItem key={index} doc={doc} index={index} />
    ));
  };

  return (
    <div className="comparison-container">
      <div className="rag-section traditional-rag">
        <h2>
          <span className="icon">🔍</span> Traditional RAG
        </h2>
        {traditional_results ? (
          <>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-value">
                  {traditional_results.retrieval_time?.toFixed(3) || 'N/A'}
                </div>
                <div className="metric-label">Time (s)</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">
                  {traditional_results.total_documents || 0}
                </div>
                <div className="metric-label">Documents</div>
              </div>
              {comparison_metrics?.score_distribution?.traditional?.mean !== undefined && (
                <div className="metric-card">
                  <div className="metric-value">
                    {comparison_metrics.score_distribution.traditional.mean.toFixed(3)}
                  </div>
                  <div className="metric-label">Avg Score</div>
                </div>
              )}
            </div>
            <div className="document-list">
              {renderDocuments(traditional_results.documents)}
            </div>
          </>
        ) : (
          <div className="loading-section">
            <div className="loading-spinner"></div>
            <p>Running traditional RAG search...</p>
          </div>
        )}
      </div>

      <div className="rag-section graph-rag">
        <h2>
          <span className="icon">🕸️</span> Graph-Enhanced RAG
        </h2>
        {graph_results ? (
          <>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-value">
                  {graph_results.retrieval_time?.toFixed(3) || 'N/A'}
                </div>
                <div className="metric-label">Time (s)</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">
                  {graph_results.total_documents || graph_results.documents?.length || 0}
                </div>
                <div className="metric-label">Documents</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">
                  {comparison_metrics?.score_distribution?.graph?.mean !== undefined && 
                   comparison_metrics.score_distribution.graph.mean !== 0 
                    ? comparison_metrics.score_distribution.graph.mean.toFixed(3) 
                    : 'N/A'}
                </div>
                <div className="metric-label">Avg Score</div>
              </div>
            </div>
            <div className="document-list">
              {renderDocuments(graph_results.documents)}
            </div>
          </>
        ) : (
          <div className="loading-section">
            <div className="loading-spinner"></div>
            <p>Running GraphRAG search...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ComparisonContainer; 