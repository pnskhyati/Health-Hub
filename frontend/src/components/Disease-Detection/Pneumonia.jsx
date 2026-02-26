import React, { useState } from 'react';
import axios from 'axios';
import './Pneumonia.css';

const Pneumonia = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Validate file type (accept DICOM, JPEG, PNG)
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/dicom'];
    const isDicom = file.name.toLowerCase().endsWith('.dcm');
    
    if (!validTypes.includes(file.type) && !isDicom) {
      setError('Please upload a valid image file (JPEG, PNG, or DICOM)');
      return;
    }
    
    setError(null);
    setImage(file);
    setResult(null);
    
    // Create preview (only for non-DICOM files)
    if (!isDicom) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setPreview(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please upload an X-ray image.');
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await axios.post(
        'http://localhost:5001/api/disease-detection', 
        formData, 
        {
          headers: {'Content-Type': 'multipart/form-data'},
          timeout: 30000 // 30 second timeout
        }
      );
      setResult(response.data);
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. Please try again.');
      } else if (err.response?.data?.error) {
        setError(`Error: ${err.response.data.error}`);
      } else {
        setError('Error detecting pneumonia. Please check if the backend server is running.');
      }
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setImage(null);
    setResult(null);
    setPreview(null);
    setError(null);
  };

  return (
    <div className="disease-detection-container">
      <div className="header">
        <h1>🫁 Pneumonia Detection from Chest X-rays</h1>
        <p className="subtitle">AI-powered medical image analysis using deep learning</p>
      </div>
      
      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="disease-form">
        <div className="form-group">
          <label htmlFor="file-input" className="file-label">
            📁 Upload Chest X-ray Image:
          </label>
          <input 
            id="file-input"
            type="file" 
            accept="image/*,.dcm"
            onChange={handleFileChange} 
            className="file-input" 
          />
          <p className="file-info">Supported formats: JPEG, PNG, DICOM (.dcm)</p>
        </div>

        <div className="button-group">
          <button 
            type="submit" 
            className="submit-button"
            disabled={loading || !image}
          >
            {loading ? (
              <>
                <span className="loading-spinner"></span>
                Analyzing...
              </>
            ) : (
              '🔍 Analyze X-ray'
            )}
          </button>
          
          {(image || result) && (
            <button 
              type="button" 
              onClick={resetForm}
              className="reset-button"
              disabled={loading}
            >
              🔄 Reset
            </button>
          )}
        </div>
      </form>

      {preview && (
        <div className="image-preview">
          <h3>📸 Uploaded X-ray Preview:</h3>
          <img src={preview} alt="X-ray preview" className="preview-image" />
        </div>
      )}

      {loading && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Processing your X-ray image... This may take a few moments.</p>
        </div>
      )}

      {result && (
        <div className={`result-container ${result.disease === 'Pneumonia' ? 'pneumonia' : 'normal'}`}>
          <div className="result-header">
            <h2>🔬 Analysis Results</h2>
            <div className={`diagnosis-badge ${result.disease.toLowerCase()}`}>
              {result.disease === 'Pneumonia' ? '🦠' : '✅'} {result.disease}
            </div>
          </div>

          <div className="result-stats">
            <div className="stat-item">
              <span className="stat-label">Diagnosis:</span>
              <span className={`stat-value ${result.disease.toLowerCase()}`}>
                {result.disease}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Confidence:</span>
              <span className="stat-value">{result.probability}%</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Model Score:</span>
              <span className="stat-value">{result.raw_probability}</span>
            </div>
          </div>
          
          <div className="image-comparison">
            <div className="image-column">
              <h3>📋 Original X-ray</h3>
              <img 
                src={`data:image/png;base64,${result.original_image}`} 
                alt="Original X-ray" 
                className="result-image"
              />
            </div>
            <div className="image-column">
              <h3>🎯 AI Attention Map</h3>
              <img 
                src={`data:image/png;base64,${result.heatmap}`} 
                alt="Pneumonia heatmap" 
                className="result-image"
              />
              <p className="heatmap-caption">
                🔥 Colored areas show where the AI model detected potential abnormalities
              </p>
            </div>
          </div>

          {result.disease === 'Pneumonia' && (
            <div className="recommendation pneumonia-warning">
              <h3>⚕️ Medical Recommendations:</h3>
              <div className="warning-text">
                <strong>⚠️ Important:</strong> This AI analysis suggests possible pneumonia. 
                Please consult with a healthcare professional immediately.
              </div>
              <ul className="recommendation-list">
                <li>🏥 Seek immediate medical attention</li>
                <li>👨‍⚕️ Consult a pulmonologist or radiologist</li>
                <li>🔄 Consider follow-up imaging</li>
                <li>💊 Discuss appropriate treatment options</li>
              </ul>
            </div>
          )}

          {result.disease === 'Normal' && (
            <div className="recommendation normal-result">
              <h3>✅ Good News!</h3>
              <p>The AI analysis suggests no signs of pneumonia in this X-ray. However, this is just a screening tool.</p>
              <ul className="recommendation-list">
                <li>🩺 Regular check-ups are still important</li>
                <li>👨‍⚕️ Consult your doctor if you have symptoms</li>
                <li>📋 Keep this analysis for your medical records</li>
              </ul>
            </div>
          )}

          <div className="disclaimer">
            <p><strong>⚠️ Medical Disclaimer:</strong> This AI tool is for educational and screening purposes only. 
            It should not replace professional medical diagnosis. Always consult qualified healthcare professionals 
            for medical advice and treatment.</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Pneumonia;