import React, { useState } from 'react';

const QuerySection = ({ onRunComparison, onRunBatchComparison }) => {
  const [query, setQuery] = useState("What are the main countries in Europe?");

  const handleSubmit = (e) => {
    e.preventDefault();
    onRunComparison(query);
  };

  const setPresetQuery = (presetQuery) => {
    setQuery(presetQuery);
  };

  return (
    <div className="query-section">
      <form onSubmit={handleSubmit} className="query-input">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your search query..."
        />
        <button type="submit" className="btn btn-primary">
          🚀 Compare
        </button>
        <button
          type="button"
          onClick={onRunBatchComparison}
          className="btn btn-secondary"
        >
          📊 Batch Test
        </button>
      </form>
      
      <div className="predefined-queries">
        <span style={{ fontWeight: 600, color: '#666' }}>Quick queries:</span>
        <span
          className="query-tag"
          onClick={() => setPresetQuery("top destinations in europe")}
        >
          European destinations
        </span>
        <span
          className="query-tag"
          onClick={() => setPresetQuery("Which countries border both Germany and France?")}
        >
          Bordering countries
        </span>
        <span
          className="query-tag"
          onClick={() => setPresetQuery("What are the cultural similarities between European nations?")}
        >
          European culture
        </span>
        <span
          className="query-tag"
          onClick={() => setPresetQuery("Describe the political systems across Europe")}
        >
          European politics
        </span>
      </div>
    </div>
  );
};

export default QuerySection; 