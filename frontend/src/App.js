import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { 
  Upload, 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  Shield, 
  Info,
  Loader2,
  File,
  X,
  RefreshCw
} from 'lucide-react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelInfo, setModelInfo] = useState(null);

  // Fetch model info on component mount
  React.useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/model-info`);
      setModelInfo(response.data);
    } catch (err) {
      console.log('Could not fetch model info');
    }
  };

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      handleFileUpload(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt'],
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    },
    multiple: false,
    maxSize: 16 * 1024 * 1024 // 16MB
  });

  const handleFileUpload = async (file) => {
    setLoading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      console.error('Error uploading file:', err);
      setError(err.response?.data?.error || 'An error occurred while analyzing the file');
    } finally {
      setLoading(false);
    }
  };

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        text: text
      });

      setResult(response.data);
    } catch (err) {
      console.error('Error analyzing text:', err);
      setError(err.response?.data?.error || 'An error occurred while analyzing the text');
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setResult(null);
    setError('');
    setText('');
  };

  const getPredictionColor = (prediction) => {
    return prediction === 'Fake' ? '#ef4444' : '#10b981';
  };

  const getPredictionIcon = (prediction) => {
    return prediction === 'Fake' ? <AlertTriangle size={24} /> : <CheckCircle size={24} />;
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Shield className="logo-icon" />
            <h1>AI Job Offer Detector</h1>
          </div>
          <p className="subtitle">Detect fake job offers using advanced AI technology</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">
          {/* Model Info */}
          {modelInfo && (
            <div className="model-info">
              <Info size={16} />
              <span>
                Model Accuracy: <strong>{modelInfo.accuracy ? (modelInfo.accuracy * 100).toFixed(1) : 'N/A'}%</strong>
                {modelInfo.total_samples && ` | Trained on ${modelInfo.total_samples.toLocaleString()} samples`}
              </span>
            </div>
          )}

          {/* Input Methods */}
          <div className="input-methods">
            {/* File Upload */}
            <div className="input-section">
              <h2>
                <Upload size={20} />
                Upload File
              </h2>
              <p>Upload PDF, DOCX, TXT, or image files</p>
              
              <div 
                {...getRootProps()} 
                className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
              >
                <input {...getInputProps()} />
                <Upload size={48} className="upload-icon" />
                <p>
                  {isDragActive
                    ? "Drop the file here..."
                    : "Drag & drop a file here, or click to select"
                  }
                </p>
                <small>Supports: PDF, DOCX, TXT, JPG, PNG, BMP, TIFF (Max: 16MB)</small>
              </div>
            </div>

            {/* Text Input */}
            <div className="input-section">
              <h2>
                <FileText size={20} />
                Paste Text
              </h2>
              <p>Or paste job offer text directly</p>
              
              <form onSubmit={handleTextSubmit}>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste your job offer letter here..."
                  rows="8"
                  className="text-input"
                />
                <button 
                  type="submit" 
                  className="analyze-btn"
                  disabled={loading || !text.trim()}
                >
                  {loading ? (
                    <>
                      <Loader2 size={16} className="spinner" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Shield size={16} />
                      Analyze Text
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="error-message">
              <X size={16} />
              <p>{error}</p>
            </div>
          )}

          {/* Loading Indicator */}
          {loading && (
            <div className="loading">
              <Loader2 size={48} className="spinner" />
              <p>Analyzing your job offer...</p>
              <small>This may take a few seconds</small>
            </div>
          )}

          {/* Results */}
          {result && (
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
                  <div 
                    className="prediction-badge"
                    style={{ backgroundColor: getPredictionColor(result.prediction) }}
                  >
                    {getPredictionIcon(result.prediction)}
                    <span>{result.prediction}</span>
                  </div>
                  
                  <div className="confidence-section">
                    <h3>Confidence Score</h3>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill"
                        style={{ 
                          width: `${result.confidence * 100}%`,
                          backgroundColor: getPredictionColor(result.prediction)
                        }}
                      />
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
                      <div key={index} className="explanation-item">
                        {item}
                      </div>
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
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Powered by Machine Learning • Built with React & Flask</p>
        <p>Protect yourself from job scams with AI-powered detection</p>
      </footer>
    </div>
  );
}

export default App; 