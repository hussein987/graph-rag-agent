import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Header from './components/Header';
import QuerySection from './components/QuerySection';
import ComparisonStats from './components/ComparisonStats';
import ChartContainer from './components/ChartContainer';
import ComparisonContainer from './components/ComparisonContainer';
import LoadingIndicator from './components/LoadingIndicator';
import './App.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [liveResults, setLiveResults] = useState({
    traditional: null,
    local: null,
    global: null
  });
  const [loadingStatus, setLoadingStatus] = useState({
    traditional: false,
    local: false,
    global: false
  });

  // Define mock data generation functions first


  const runParallelComparison = useCallback(async (query) => {
    if (!query.trim()) {
      alert('Please enter a search query');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    setLiveResults({
      traditional: null,
      local: null,
      global: null
    });
    setLoadingStatus({
      traditional: true,
      local: true,
      global: true
    });

    const requestBody = { query: query, k: 5 };

    // Function to call an endpoint and update results
    const callEndpoint = async (endpoint, resultKey) => {
      try {
        const response = await axios.post(`/api/${endpoint}`, requestBody);
        
        setLiveResults(prev => ({
          ...prev,
          [resultKey]: response.data
        }));
        
        setLoadingStatus(prev => ({
          ...prev,
          [resultKey]: false
        }));
        
      } catch (error) {
        console.error(`Error calling ${endpoint}:`, error);
        
        setLiveResults(prev => ({
          ...prev,
          [resultKey]: {
            error: error.response?.data?.error || `${endpoint} failed`,
            method: `${endpoint}_error`,
            retrieval_time: 0,
            total_documents: 0,
            strategy: `Error - ${endpoint} failed`,
            documents: [],
            query: query,
            timestamp: Date.now()
          }
        }));
        
        setLoadingStatus(prev => ({
          ...prev,
          [resultKey]: false
        }));
      }
    };

    // Call all endpoints simultaneously
    const promises = [
      callEndpoint('traditional-rag', 'traditional'),
      callEndpoint('local-search', 'local'),
      callEndpoint('global-search', 'global')
    ];

    try {
      await Promise.all(promises);
      setLoading(false);
    } catch (error) {
      console.error('Error in parallel comparison:', error);
      setError('Some endpoints failed');
      setLoading(false);
    }
  }, []);



  // Create combined results for display
  const getCombinedResults = () => {
    const { traditional, local, global } = liveResults;
    
    // Only show results if we have at least traditional results to prevent null errors
    if (!traditional) return null;
    
    // Combine local and global into graph_results
    const graphDocuments = [];
    let graphRetrievalTime = 0;
    let graphStrategy = "GraphRAG Search";
    
    if (local && local.documents) {
      graphDocuments.push(...local.documents);
      graphRetrievalTime += local.retrieval_time || 0;
    }
    
    if (global && global.documents) {
      graphDocuments.push(...global.documents);
      graphRetrievalTime += global.retrieval_time || 0;
    }
    
    if (local && global) {
      graphStrategy = "GraphRAG (Local + Global Search)";
    } else if (local) {
      graphStrategy = "GraphRAG (Local Search Only)";
    } else if (global) {
      graphStrategy = "GraphRAG (Global Search Only)";
    }
    
    const combinedResults = {
      query: traditional?.query || local?.query || global?.query || "",
      traditional_results: {
        method: traditional.method || "traditional_rag",
        retrieval_time: traditional.retrieval_time || 0,
        total_documents: traditional.total_documents || 0,
        strategy: traditional.strategy || "Traditional RAG",
        documents: traditional.documents || []
      },
      graph_results: (local || global) ? {
        method: "working_graph_rag_combined",
        retrieval_time: graphRetrievalTime,
        total_documents: graphDocuments.length,
        strategy: graphStrategy,
        documents: graphDocuments
      } : {
        method: "working_graph_rag_combined",
        retrieval_time: 0,
        total_documents: 0,
        strategy: "GraphRAG Loading...",
        documents: []
      },
      comparison_metrics: {
        retrieval_time: {
          traditional: traditional?.retrieval_time || 0,
          graph: graphRetrievalTime
        },
        document_overlap: { overlap_percentage: 0 },
        score_distribution: {
          traditional: { min: 0, mean: 0, max: 0 },
          graph: { min: 0, mean: 0, max: 0 }
        },
        graph_traversal: { success: !!(local || global) }
      }
    };
    
    return combinedResults;
  };

  useEffect(() => {
    const combinedResults = getCombinedResults();
    if (combinedResults) {
      setResults(combinedResults);
    }
  }, [liveResults]);

  return (
    <div className="App">
      <div className="container">
        <Header />
        
        <QuerySection 
          onRunComparison={runParallelComparison}
        />

        {loading && (
          <div className="loading-container">
            <LoadingIndicator />
            <div className="loading-status">
              <div className="status-grid">
                <div className={`status-item ${loadingStatus.traditional ? 'loading' : 'complete'}`}>
                  <span className="status-icon">
                    {loadingStatus.traditional ? '⏳' : '✅'}
                  </span>
                  <span>Traditional RAG</span>
                </div>
                <div className={`status-item ${loadingStatus.local ? 'loading' : 'complete'}`}>
                  <span className="status-icon">
                    {loadingStatus.local ? '⏳' : '✅'}
                  </span>
                  <span>Local Search</span>
                </div>
                <div className={`status-item ${loadingStatus.global ? 'loading' : 'complete'}`}>
                  <span className="status-icon">
                    {loadingStatus.global ? '⏳' : '✅'}
                  </span>
                  <span>Global Search</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {results && (
          <>
            <ComparisonStats results={results} />
            <ChartContainer results={results} />
            <ComparisonContainer results={results} />
          </>
        )}

        {error && (
          <div className="error-message">
            <h3>⚠️ Error</h3>
            <p>{error}</p>
            <p>Please try again or check the server connection.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App; 