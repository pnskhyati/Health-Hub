// Fixed React Heart Detection Component

import React, { useState, useRef, useCallback } from 'react';
import { Heart, Upload, Activity, AlertCircle, CheckCircle, Loader, Download, Eye, EyeOff } from 'lucide-react';
import './HeartDetection.css';

const HeartDetection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = useCallback((file) => {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/dicom'];
    const isDCM = file.name.toLowerCase().endsWith('.dcm');
    const isValidImage = validTypes.includes(file.type) || 
                        file.name.toLowerCase().match(/\.(jpg|jpeg|png)$/);
    
    if (!isValidImage && !isDCM) {
      setError('Please select a valid image file (JPG, PNG) or DICOM file (.dcm)');
      return;
    }

    // Check file size (limit to 50MB)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      setError('File size too large. Please select a file smaller than 50MB.');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResults(null);

    // Create preview for non-DICOM files
    if (!isDCM) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setFilePreview({
          name: file.name,
          size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
          type: file.type || 'image',
          url: e.target.result
        });
      };
      reader.onerror = () => {
        setError('Failed to read file. Please try again.');
      };
      reader.readAsDataURL(file);
    } else {
      setFilePreview({
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        type: 'application/dicom',
        url: null
      });
    }
  }, []);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, [handleFileSelect]);

  const analyzeHeart = async () => {
    if (!selectedFile) {
      setError('Please upload an image first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

      const response = await fetch('http://localhost:5001/api/heart-detection', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorMessage;
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || `Server error: ${response.status}`;
        } catch {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      // Validate response data
      if (!data.success) {
        throw new Error(data.error || 'Analysis failed');
      }

      if (!data.bbox_coordinates || !Array.isArray(data.bbox_coordinates)) {
        throw new Error('Invalid response format: missing bbox coordinates');
      }

      setResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      
      if (err.name === 'AbortError') {
        setError('Request timed out. Please try again.');
      } else if (err.message.includes('fetch')) {
        setError('Cannot connect to server. Please ensure the Flask server is running on http://localhost:5001');
      } else {
        setError(err.message || 'Failed to analyze heart image');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const downloadReport = () => {
    if (!results) return;

    const reportData = {
      diagnosis: results.primary_condition,
      confidence: results.primary_confidence,
      allFindings: results.conditions.map((condition, index) => ({
        condition,
        confidence: results.confidence_scores[index]
      })),
      bboxCoordinates: results.bbox_coordinates,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `heart_analysis_report_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="heart-detection-container">
      <div className="header">
        <Heart className="header-icon" />
        <h1>Cardiac Analysis</h1>
        <p>Upload a chest X-ray or MRI to detect heart conditions</p>
      </div>

      {error && (
        <div className="error-message">
          <AlertCircle className="error-icon" />
          <span>{error}</span>
          <button 
            onClick={() => setError(null)}
            className="error-close"
            aria-label="Close error"
          >
            ×
          </button>
        </div>
      )}

      <div className="upload-section">
        <div 
          className={`drop-zone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.dcm"
            onChange={(e) => handleFileSelect(e.target.files?.[0])}
            style={{ display: 'none' }}
          />
          <Upload className="upload-icon" />
          <h3>Drag & Drop or Click to Upload</h3>
          <p>Supports JPG, PNG, and DICOM formats (max 50MB)</p>
        </div>

        {filePreview && (
          <div className="file-preview">
            {filePreview.url ? (
              <img 
                src={filePreview.url} 
                alt="Preview" 
                className="image-preview"
                onError={() => setError('Failed to display image preview')}
              />
            ) : (
              <div className="dicom-placeholder">
                <Activity className="dicom-icon" />
                <span>DICOM File</span>
              </div>
            )}
            <div className="file-info">
              <p className="file-name">{filePreview.name}</p>
              <p className="file-details">{filePreview.size} • {filePreview.type}</p>
            </div>
            <CheckCircle className="check-icon" />
          </div>
        )}

        <button
          onClick={analyzeHeart}
          disabled={isAnalyzing || !selectedFile}
          className="analyze-button"
        >
          {isAnalyzing ? (
            <>
              <Loader className="spinner" />
              Analyzing...
            </>
          ) : (
            <>
              <Activity className="analyze-icon" />
              Analyze Heart
            </>
          )}
        </button>
      </div>

      {results && (
        <div className="results-section">
          <h2>
            <Heart className="results-icon" />
            Analysis Results
          </h2>

          <div className="diagnosis-card">
            <h3>Primary Diagnosis</h3>
            <p className="diagnosis">{results.primary_condition}</p>
            <div className="confidence">
              <span>Confidence: {results.primary_confidence}%</span>
            </div>
          </div>

          <div className="visualization-tabs">
            <button
              className={`tab ${!showHeatmap ? 'active' : ''}`}
              onClick={() => setShowHeatmap(false)}
            >
              <Eye className="tab-icon" />
              Bounding Box
            </button>
            <button
              className={`tab ${showHeatmap ? 'active' : ''}`}
              onClick={() => setShowHeatmap(true)}
            >
              <EyeOff className="tab-icon" />
              Heatmap View
            </button>
          </div>

          <div className="image-container">
            {showHeatmap ? (
              <img 
                src={`data:image/png;base64,${results.heatmap}`} 
                alt="Heart heatmap" 
                className="result-image"
                onError={() => setError('Failed to display heatmap')}
              />
            ) : (
              <img 
                src={`data:image/png;base64,${results.bbox_image}`} 
                alt="Heart with bounding box" 
                className="result-image"
                onError={() => setError('Failed to display bounding box image')}
              />
            )}
          </div>

          <div className="additional-findings">
            <h3>Additional Findings</h3>
            <ul>
              {results.conditions.map((condition, index) => (
                <li key={index}>
                  <span className="condition">{condition}</span>
                  <span className="confidence-badge">
                    {results.confidence_scores[index]}% confidence
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <button className="download-button" onClick={downloadReport}>
            <Download className="download-icon" />
            Download Full Report
          </button>
        </div>
      )}
    </div>
  );
};

export default HeartDetection;