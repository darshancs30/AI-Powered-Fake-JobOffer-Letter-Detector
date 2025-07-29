import React from 'react';
import { X } from 'lucide-react';
import '../App.css';

const ErrorMessage = ({ error }) => (
  error ? (
    <div className="error-message">
      <X size={16} />
      <p>{error}</p>
    </div>
  ) : null
);

export default ErrorMessage;
