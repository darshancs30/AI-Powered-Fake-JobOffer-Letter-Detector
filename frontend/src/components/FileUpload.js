import React from 'react';
import { Upload } from 'lucide-react';
import '../App.css';


const FileUpload = ({ getRootProps, getInputProps, isDragActive }) => (
  <div className="input-section">
    <h2>
      <Upload size={20} />
      Upload File
      <img src="https://cdn-icons-png.flaticon.com/512/3062/3062635.png" alt="Fraud Alert" style={{ width: '32px', height: '32px', marginLeft: '8px', animation: 'floatY 2s infinite alternate' }} />
    </h2>
    <p>Upload PDF, DOCX, TXT, or image files</p>
    <div {...getRootProps()} className={`dropzone ${isDragActive ? 'drag-active' : ''}`}>
      <input {...getInputProps()} />
      <Upload size={48} className="upload-icon" />
      <p>{isDragActive ? "Drop the file here..." : "Drag & drop a file here, or click to select"}</p>
      <small>Supports: PDF, DOCX, TXT, JPG, PNG, BMP, TIFF (Max: 16MB)</small>
    </div>
  </div>
);

export default FileUpload;
