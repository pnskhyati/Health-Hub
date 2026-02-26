import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './DiseaseDetection.css';

const DiseaseDetection = () => {
  const [selectedOption, setSelectedOption] = useState(null);
  const navigate = useNavigate();

  const diseaseOptions = [
    {
      id: 'pneumonia',
      name: 'Pneumonia Detection',
      description: 'Analyze chest X-rays for signs of pneumonia',
      icon: '🫁'
    },
    {
      id: 'atrium-segmentation',
      name: 'Atrium Segmentation',
      description: 'Detailed analysis of heart atrium structures',
      icon: '❤️'
    },
    {
      id: 'heart-detection',
      name: 'Heart Detection',
      description: 'Comprehensive cardiac imaging analysis',
      icon: '💓'
    }
  ];

  const handleSelect = (option) => {
    setSelectedOption(option);
    navigate(`/analyze/${option.id}`);
  };

  return (
    <div className="disease-container">
      <header className="disease-header">
        <h1 className="disease-title">Medical Imaging Analysis</h1>
        <p className="disease-subtitle">Select the diagnostic procedure you need</p>
      </header>
      
      <div className="options-grid">
        {diseaseOptions.map((option) => (
          <div 
            key={option.id}
            className={`option-card ${selectedOption?.id === option.id ? 'selected' : ''}`}
            onClick={() => handleSelect(option)}
          >
            <div className="option-icon">{option.icon}</div>
            <h2 className="option-name">{option.name}</h2>
            <p className="option-description">{option.description}</p>
            <div className="select-indicator">▶</div>
          </div>
        ))}
      </div>
      
      <div className="gradient-footer">
        <p className="footer-text">Advanced AI-powered medical diagnostics</p>
      </div>
    </div>
  );
};

export default DiseaseDetection;