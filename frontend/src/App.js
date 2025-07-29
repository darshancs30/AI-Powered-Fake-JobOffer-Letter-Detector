import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';
import Header from './components/Header';
import { AlertTriangle, CheckCircle } from 'lucide-react';
import Footer from './components/Footer';
import ModelInfo from './components/ModelInfo';
import ErrorMessage from './components/ErrorMessage';
import Loading from './components/Loading';
import FileUpload from './components/FileUpload';
import TextInput from './components/TextInput';
import Results from './components/Results';

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


  // Animation hooks for entrance effects
  React.useEffect(() => {
    const app = document.querySelector('.App');
    if (app) app.classList.add('app-animate');
  }, []);

  return (
    <div className="App app-animate" style={{ position: 'relative', minHeight: '100vh', overflow: 'hidden' }}>

      {/* Animated AI/Fraud Background */}
      <div className="ai-bg-animation bg-float"></div>
      <div className="ai-bg-images">
        <img src="https://cdn-icons-png.flaticon.com/512/3062/3062634.png" alt="AI Shield" className="img1 spin" />
        <img src="https://cdn-icons-png.flaticon.com/512/3062/3062635.png" alt="Fraud Alert" className="img2 pulse" />
        <img src="https://cdn-icons-png.flaticon.com/512/3062/3062636.png" alt="AI Brain" className="img3 float" />
      </div>

      {/* Header with entrance animation */}
      <div className="header-animate">
        <Header />
      </div>

      {/* Main Content with fade/slide animation */}
      <main className="main-content fade-in" style={{ position: 'relative', zIndex: 2 }}>
        <div className="container scale-in">
          <ModelInfo modelInfo={modelInfo} />
          <div className="input-methods input-animate">
            <FileUpload getRootProps={getRootProps} getInputProps={getInputProps} isDragActive={isDragActive} />
            <TextInput text={text} setText={setText} handleTextSubmit={handleTextSubmit} loading={loading} />
          </div>
          <ErrorMessage error={error} animate />
          <Loading loading={loading} animate />
          <Results result={result} clearResults={clearResults} animate />
        </div>
      </main>

      {/* Footer with entrance animation */}
      <div className="footer-animate">
        <Footer />
      </div>
    </div>
  );
}

export default App;