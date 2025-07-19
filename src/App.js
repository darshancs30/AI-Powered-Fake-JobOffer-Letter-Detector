import React, { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, XCircle, Loader, FileText, Upload } from 'lucide-react';
import './App.css';
import Tesseract from 'tesseract.js';
import { getDocument, GlobalWorkerOptions, version as pdfjsVersion } from 'pdfjs-dist';
GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsVersion}/pdf.worker.min.js`;

function App() {
  const [file, setFile] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [extracting, setExtracting] = useState(false);
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');

  // Handle file selection and extraction
  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setExtractedText('');
    setError('');
    setResult(null);
    if (!selectedFile) return;
    setExtracting(true);
    try {
      const ext = selectedFile.name.split('.').pop().toLowerCase();
      let text = '';
      if (ext === 'pdf') {
        // PDF extraction
        const arrayBuffer = await selectedFile.arrayBuffer();
        const pdf = await getDocument({ data: arrayBuffer }).promise;
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          text += content.items.map(item => item.str).join(' ') + '\n';
        }
      } else if (['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'].includes(ext)) {
        // Image OCR
        const imageUrl = URL.createObjectURL(selectedFile);
        const { data: { text: ocrText } } = await Tesseract.recognize(imageUrl, 'eng');
        text = ocrText;
        URL.revokeObjectURL(imageUrl);
      } else if (ext === 'docx') {
        // DOCX extraction placeholder
        text = 'DOCX extraction is not supported in-browser. Please use PDF or image, or integrate backend extraction.';
      } else {
        text = 'Unsupported file type. Please upload a PDF, image, or DOCX file.';
      }
      setExtractedText(text);
      setExtracting(false);
      // Immediately analyze after extraction
      if (text && !text.startsWith('Unsupported') && !text.startsWith('DOCX extraction')) {
        analyzeText(text);
      }
    } catch (err) {
      setExtractedText('Failed to extract text from file.');
      setExtracting(false);
    }
  };

  // Analyze extracted text (mocked)
  const analyzeText = async (text) => {
    setAnalyzing(true);
    setResult(null);
    setError('');
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      // Mock result
      const mockResult = {
        prediction: Math.random() > 0.5 ? 'legitimate' : 'fraudulent',
        confidence: Math.floor(Math.random() * 40) + 60,
        riskFactors: [
          'Unusual salary range for the position',
          'Generic job description',
          'Poor grammar in communication'
        ],
        recommendations: [
          'Verify company information through official channels',
          'Check for company reviews on Glassdoor or LinkedIn',
          'Never share personal financial information'
        ]
      };
      setResult(mockResult);
    } catch (err) {
      setError('An error occurred while analyzing the job offer. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  const getResultIcon = (prediction) => {
    switch (prediction) {
      case 'legitimate':
        return <CheckCircle size={48} color="#28a745" />;
      case 'fraudulent':
        return <XCircle size={48} color="#dc3545" />;
      case 'suspicious':
        return <AlertTriangle size={48} color="#ffc107" />;
      default:
        return <AlertTriangle size={48} color="#6c757d" />;
    }
  };

  const getResultClass = (prediction) => {
    switch (prediction) {
      case 'legitimate':
        return 'legitimate';
      case 'fraudulent':
        return 'fraudulent';
      case 'suspicious':
        return 'suspicious';
      default:
        return '';
    }
  };

  const getResultTitle = (prediction) => {
    switch (prediction) {
      case 'legitimate':
        return 'Likely Legitimate';
      case 'fraudulent':
        return 'Potential Fraud';
      case 'suspicious':
        return 'Suspicious';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="App">
      {/* Animated AI/Tech SVG Background */}
      <div className="background-hero">
        <svg viewBox="0 0 1440 900" fill="none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="ai1" x1="0" y1="0" x2="1440" y2="900" gradientUnits="userSpaceOnUse">
              <stop stopColor="#00c3ff" />
              <stop offset="1" stopColor="#4e54c8" />
            </linearGradient>
          </defs>
          <circle cx="1200" cy="200" r="180" fill="url(#ai1)" opacity="0.18" />
          <circle cx="300" cy="700" r="220" fill="url(#ai1)" opacity="0.13" />
          <g opacity="0.12">
            <rect x="200" y="100" width="1040" height="700" rx="80" fill="#fff" />
            <path d="M400 200 Q600 400 800 200 T1200 200" stroke="#fff" strokeWidth="4" fill="none" />
            <path d="M400 700 Q600 500 800 700 T1200 700" stroke="#fff" strokeWidth="4" fill="none" />
            <circle cx="720" cy="450" r="60" fill="#fff" />
            <circle cx="720" cy="450" r="30" fill="#00c3ff" opacity="0.3" />
          </g>
        </svg>
      </div>
      <div className="container">
        <header className="header">
          <div className="header-content">
            <Shield size={48} color="white" />
            <h1>AI Powered Fake Job/Internship Offer Letter Detector</h1>
            <p>Upload a job or internship offer document (PDF, image, DOCX) to analyze its legitimacy using AI.</p>
          </div>
        </header>

        <main className="main-content">
          <div className="form-section">
            <div className="card">
              <h2>Upload & Analyze</h2>
              <div className="form-group full-width">
                <label className="form-label">
                  <Upload size={20} style={{ marginRight: 8, verticalAlign: 'middle' }} />
                  Upload Offer Letter (PDF, Image, DOCX)
                </label>
                <input
                  type="file"
                  accept=".pdf,.docx,.jpg,.jpeg,.png,.bmp,.gif,.webp"
                  onChange={handleFileChange}
                  className="form-input"
                />
                {extracting && (
                  <div className="loading" style={{ padding: 10 }}>
                    <div className="spinner"></div>
                    <span style={{ marginLeft: 12 }}>Extracting text...</span>
                  </div>
                )}
                {extractedText && (
                  <div className="card" style={{ marginTop: 16, background: '#f8f9fa' }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                      <FileText size={20} style={{ marginRight: 8 }} />
                      <strong>Extracted Text</strong>
                    </div>
                    <textarea
                      className="form-input form-textarea"
                      style={{ minHeight: 100 }}
                      value={extractedText}
                      readOnly
                    />
                  </div>
                )}
              </div>
            </div>
          </div>

          {error && (
            <div className="alert alert-danger">
              {error}
            </div>
          )}

          {analyzing && (
            <div className="card">
              <div className="loading">
                <div className="spinner"></div>
                <p style={{ marginLeft: '16px' }}>Analyzing job offer...</p>
              </div>
            </div>
          )}

          {result && (
            <div className="result-section">
              <div className={`card result-card ${getResultClass(result.prediction)}`}>
                <div className="result-header">
                  {getResultIcon(result.prediction)}
                  <div className="result-info">
                    <h3>{getResultTitle(result.prediction)}</h3>
                    <p>Confidence: {result.confidence}%</p>
                    <div className="confidence-bar">
                      <div
                        className={`confidence-fill ${getResultClass(result.prediction)}`}
                        style={{ width: `${result.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
                {result.riskFactors && result.riskFactors.length > 0 && (
                  <div className="result-section">
                    <h4>Risk Factors Detected:</h4>
                    <ul className="risk-factors">
                      {result.riskFactors.map((factor, index) => (
                        <li key={index}>{factor}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.recommendations && result.recommendations.length > 0 && (
                  <div className="result-section">
                    <h4>Recommendations:</h4>
                    <ul className="recommendations">
                      {result.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
        <footer className="footer">
          <p>&copy; 2024 AI Powered Fake Job/Internship Offer Letter Detector. Built with AI to protect job seekers.</p>
        </footer>
      </div>
    </div>
  );
}

export default App; 