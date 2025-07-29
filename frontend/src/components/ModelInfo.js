import React from 'react';
import { Info } from 'lucide-react';
import '../App.css';


const ModelInfo = ({ modelInfo }) => (
  modelInfo ? (
    <div className="model-info">
      <Info size={16} />
      <span>
        Model Accuracy: <strong>{modelInfo.accuracy ? (modelInfo.accuracy * 100).toFixed(1) : 'N/A'}%</strong>
        {modelInfo.total_samples && ` | Trained on ${modelInfo.total_samples.toLocaleString()} samples`}
        <img src="https://cdn-icons-png.flaticon.com/512/3062/3062636.png" alt="AI Brain" style={{ width: '22px', height: '22px', marginLeft: '8px', animation: 'floatY 2.2s infinite alternate' }} />
      </span>
    </div>
  ) : null
);

export default ModelInfo;
