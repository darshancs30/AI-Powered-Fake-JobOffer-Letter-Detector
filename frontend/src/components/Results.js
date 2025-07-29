import React from 'react';
import { RefreshCw, File, AlertTriangle, CheckCircle } from 'lucide-react';
import '../App.css';

const getPredictionColor = (prediction) => prediction === 'Fake' ? '#ef4444' : '#10b981';
const getPredictionIcon = (prediction) => prediction === 'Fake' ? <AlertTriangle size={24} /> : <CheckCircle size={24} />;

const Results = ({ result, clearResults }) => (
  result ? (
    <div className="results">
      <div className="results-header">
        <h2>Analysis Results</h2>
        <button onClick={clearResults} className="clear-btn">
          <RefreshCw size={16} />
          Analyze Another
        </button>
      </div>
      <div className="result-card">
        {/* Prediction */}
        <div className="prediction-section">
          <div className="prediction-badge" style={{ backgroundColor: getPredictionColor(result.prediction) }}>
            {getPredictionIcon(result.prediction)}
            <span>{result.prediction}</span>
          </div>
          <div className="confidence-section">
            <h3>Confidence Score</h3>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: `${result.confidence * 100}%`, backgroundColor: getPredictionColor(result.prediction) }} />
            </div>
            <p className="confidence-text">{result.confidence_percentage}</p>
          </div>
        </div>
        {/* File Info */}
        {result.filename && (
          <div className="file-info">
            <File size={16} />
            <span>
              <strong>{result.filename}</strong> ({result.file_type.toUpperCase()})
            </span>
          </div>
        )}
        {/* Fraud Indicators */}
        <div className="indicators-section">
          <h3>Fraud Indicators Detected</h3>
          <div className="indicators-grid">
            {Object.entries(result.fraud_indicators).map(([key, value]) => (
              <div key={key} className={`indicator ${value > 0 ? 'detected' : 'none'}`}>
                <span className="indicator-name">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </span>
                <span className="indicator-value">
                  {value > 0 ? `${value} detected` : 'None found'}
                </span>
              </div>
            ))}
          </div>
        </div>
        {/* Explanation */}
        <div className="explanation-section">
          <h3>AI Explanation</h3>
          <div className="explanation-list">
            {result.explanation.map((item, index) => (
              <div key={index} className="explanation-item">{item}</div>
            ))}
          </div>
        </div>
        {/* Text Preview */}
        <div className="text-preview">
          <h3>Extracted Text Preview</h3>
          <div className="text-content">
            {result.text}
            {result.text_length > 500 && (
              <div className="text-truncated">
                <small>... (showing first 500 characters)</small>
              </div>
            )}
          </div>
          <small>Text length: {result.text_length} characters</small>
        </div>
      </div>
    </div>
  ) : null
);

export default Results;
