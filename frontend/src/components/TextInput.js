import React from 'react';
import { FileText, Shield, Loader2 } from 'lucide-react';
import '../App.css';

const TextInput = ({ text, setText, handleTextSubmit, loading }) => (
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
      <button type="submit" className="analyze-btn" disabled={loading || !text.trim()}>
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
);

export default TextInput;
