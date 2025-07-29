import React from 'react';
import { Loader2 } from 'lucide-react';
import '../App.css';

const Loading = ({ loading }) => (
  loading ? (
    <div className="loading">
      <Loader2 size={48} className="spinner" />
      <p>Analyzing your job offer...</p>
      <small>This may take a few seconds</small>
    </div>
  ) : null
);

export default Loading;
