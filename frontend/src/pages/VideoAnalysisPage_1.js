import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as videoService from '../services/videoService';
import './VideoAnalysisPage.css';
import { 
  Box, 
  Typography
} from '@mui/material';

const API_URL = `${process.env.REACT_APP_API_URL}` || 'https://3ed0-103-88-237-251.ngrok-free.app'; 

function VideoAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [classificationError, setClassificationError] = useState('');
  const [isClassifying, setIsClassifying] = useState(false);
  const [captionResult, setCaptionResult] = useState(null);
  const [captionError, setCaptionError] = useState('');
  const [isCaptioning, setIsCaptioning] = useState(false);
  const [displayedCaption, setDisplayedCaption] = useState('');
  const [classificationLoadingMessage, setClassificationLoadingMessage] = useState('');
  const [captioningLoadingMessage, setCaptioningLoadingMessage] = useState('');
  const [evidenceExtractionResult, setEvidenceExtractionResult] = useState(null);
  const [isExtractingEvidence, setIsExtractingEvidence] = useState(false);
  const [evidenceError, setEvidenceError] = useState('');
  const [evidenceLoadingMessage, setEvidenceLoadingMessage] = useState('');
  const evidenceResultsRef = useRef(null);

  const classificationMessages = useMemo(() => [
    "UPLOADING AND PREPARING VIDEO...",
    "STAGE 1: DETECTING POTENTIAL CRIME FRAMES (CLIP)...",
    "STAGE 2: IDENTIFYING EVENT SEGMENTS...",
    "STAGE 3: CLASSIFYING SEGMENT CRIME TYPES (TIMESFORMER)...",
    "COMPILING FINAL REPORT..."
  ], []);

  const captioningMessages = useMemo(() => [
    "CONNECTING TO NARRATIVE AI...",
    "UPLOADING VIDEO CONTEXT...",
    "AI IS COMPOSING DESCRIPTION...",
    "FETCHING GENERATED CAPTION..."
  ], []);

  const evidenceMessages = useMemo(() => [
    "UPLOADING VIDEO...",
    "STAGE 1: EXTRACTING FRAMES...",
    "STAGE 2: RUNNING OBJECT DETECTION...",
    "STAGE 3: ANALYZING SUSPICIOUS OBJECTS...",
    "STAGE 4: GENERATING EVIDENCE REPORT..."
  ], []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
      // Clear all previous results
      setClassificationResult(null);
      setClassificationError('');
      setCaptionResult(null);
      setCaptionError('');
      setEvidenceExtractionResult(null);
      setEvidenceError('');
      setDisplayedCaption('');
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
    }
  };

  const handleClassify = async () => {
    if (!selectedFile) {
      setClassificationError('Please select a video file first.');
      return;
    }
    setIsClassifying(true);
    setClassificationError('');
    setClassificationResult(null);
    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await videoService.uploadVideoForClassification(formData);
      if (response.success) {
        setClassificationResult(response);
      } else {
        setClassificationError(response.message || 'Classification failed.');
      }
    } catch (error) {
      console.error("Classification error:", error);
      setClassificationError(error.message || 'An unexpected error occurred during classification.');
    }
    setIsClassifying(false);
  };

  const handleCaption = async () => {
    if (!selectedFile) {
      setCaptionError('Please select a video file first.');
      return;
    }
    setIsCaptioning(true);
    setCaptionError('');
    setCaptionResult(null);
    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await videoService.uploadVideoForCaptioning(formData);
      if (response.success && response.data && response.data.caption) {
        setCaptionResult({
          caption: response.data.caption
        });
        // Scroll to caption results after a short delay
        setTimeout(() => {
          const captionElement = document.querySelector('.caption-results');
          if (captionElement) {
            captionElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 100);
      } else {
        setCaptionError(response.message || 'Captioning failed. Please try again.');
      }
    } catch (error) {
      console.error("Captioning error:", error);
      setCaptionError(error.message || 'An unexpected error occurred during captioning.');
    } finally {
      setIsCaptioning(false);
    }
  };

  const handleExtractEvidence = async () => {
    if (!selectedFile) {
      setEvidenceError('Please select a video file first.');
      return;
    }
    setIsExtractingEvidence(true);
    setEvidenceError('');
    setEvidenceExtractionResult(null);
    
    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await videoService.extractEvidence(formData);
      if (response.success) {
        setEvidenceExtractionResult(response.data);
        
        setTimeout(() => {
          if (evidenceResultsRef.current) {
            evidenceResultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 100);
      } else {
        setEvidenceError(response.message || 'Evidence extraction failed.');
      }
    } catch (error) {
      setEvidenceError(error.message || 'An unexpected error occurred during evidence extraction.');
      console.error("Evidence extraction error:", error);
    }
    setIsExtractingEvidence(false);
  };

  useEffect(() => {
    if (captionResult && captionResult.caption) {
      setDisplayedCaption(captionResult.caption.charAt(0) || ''); 
      
      let index = 1;
      const intervalId = setInterval(() => {
        if (index < captionResult.caption.length) {
          setDisplayedCaption((prev) => prev + captionResult.caption.charAt(index));
          index++;
        } else {
          clearInterval(intervalId);
        }
      }, 50);
      return () => clearInterval(intervalId);
    } else {
      setDisplayedCaption('');
    }
  }, [captionResult]);

  useEffect(() => {
    let intervalId;
    if (isClassifying) {
      let msgIndex = 0;
      setClassificationLoadingMessage(classificationMessages[msgIndex]);
      intervalId = setInterval(() => {
        msgIndex = (msgIndex + 1) % classificationMessages.length;
        setClassificationLoadingMessage(classificationMessages[msgIndex]);
      }, 1500);
    } else {
      setClassificationLoadingMessage('');
    }
    return () => clearInterval(intervalId);
  }, [isClassifying, classificationMessages]);

  useEffect(() => {
    let intervalId;
    if (isCaptioning) {
      let msgIndex = 0;
      setCaptioningLoadingMessage(captioningMessages[msgIndex]);
      intervalId = setInterval(() => {
        msgIndex = (msgIndex + 1) % captioningMessages.length;
        setCaptioningLoadingMessage(captioningMessages[msgIndex]);
      }, 1500);
    } else {
      setCaptioningLoadingMessage('');
    }
    return () => clearInterval(intervalId);
  }, [isCaptioning, captioningMessages]);

  useEffect(() => {
    let intervalId;
    if (isExtractingEvidence) {
      let msgIndex = 0;
      setEvidenceLoadingMessage(evidenceMessages[msgIndex]);
      intervalId = setInterval(() => {
        msgIndex = (msgIndex + 1) % evidenceMessages.length;
        setEvidenceLoadingMessage(evidenceMessages[msgIndex]);
      }, 1500);
    } else {
      setEvidenceLoadingMessage('');
    }
    return () => clearInterval(intervalId);
  }, [isExtractingEvidence, evidenceMessages]);

  return (
    <div className="video-analysis-page">
      <h2>Crime Detection</h2>
      
      <div className="upload-section styled-box">
        <label htmlFor="video-upload">UPLOAD VIDEO_FILE:</label>
        <input 
          type="file" 
          id="video-upload" 
          className="custom-file-input"
          accept="video/mp4,video/avi,video/mov,video/mkv" 
          onChange={handleFileChange} 
        />
        {previewUrl && selectedFile && (
          <div className="video-preview-container">
            <h3>PREVIEW:</h3>
            <video controls src={previewUrl} style={{maxWidth: '100%', height: 'auto'}} >
              Your browser does not support the video tag.
            </video>
            <p className="filename-display">{selectedFile.name}</p>
          </div>
        )}
      </div>

      <div className="actions-section">
        <button onClick={handleClassify} className="pixel-button" disabled={!selectedFile || isClassifying || isCaptioning || isExtractingEvidence}>
          {isClassifying ? 'CLASSIFYING...' : 'CLASSIFY_SCENE'}
        </button>
        <button onClick={handleCaption} className="pixel-button" disabled={!selectedFile || isCaptioning || isClassifying || isExtractingEvidence}>
          {isCaptioning ? 'CAPTIONING...' : 'GENERATE_CAPTION'}
        </button>
        <button onClick={handleExtractEvidence} className="pixel-button" disabled={!selectedFile || isExtractingEvidence || isClassifying || isCaptioning}>
          {isExtractingEvidence ? 'EXTRACTING...' : 'EXTRACT_EVIDENCE'}
        </button>
      </div>

      {(isClassifying || isCaptioning || isExtractingEvidence) && (
        <div className="loading-indicator-container styled-box"> 
          <div className="loading-text pixel-font">
            {isClassifying && <p>{classificationLoadingMessage}</p>}
            {isCaptioning && <p>{captioningLoadingMessage}</p>}
            {isExtractingEvidence && <p>{evidenceLoadingMessage}</p>}
          </div>
          <div className="pixel-progress-bar-container">
            <div className="pixel-progress-bar"></div>
          </div>
        </div>
      )}

      <div className="results-container">
        {classificationError && <p className="error-message form-error-message">Classification Error: {classificationError}</p>}
        
        {classificationResult && (
          <div className="classification-results styled-box">
            <h3>== CRIME DETECTION REPORT ==</h3>
            <p><strong>Uploaded File:</strong> {classificationResult.filename}</p>
            <p><strong>Timesformer Prediction:</strong> {classificationResult.most_common_prediction}</p>
            <p><strong>CLIP Prediction:</strong> {classificationResult.most_common_clip_prediction}</p>
            <p className="model-note"><em>Note: For old, unclear, or blurry videos, check Timesformer output. For new, high-quality videos, use CLIP output.</em></p>
            
            {classificationResult.event_segments && classificationResult.event_segments.length > 0 && (
              <div className="event-segments-results styled-box" style={{ marginLeft: 0, marginRight: 0, width: '100%' }}>
                <h4>IDENTIFIED EVENT SEGMENTS:</h4>
                {classificationResult.event_segments.map((segment, index) => (
                  <div key={`segment-${index}`} className="event-segment-item styled-box-inner">
                    <p>
                      <strong>Segment {index + 1}:</strong><br />
                      <strong>Timesformer Prediction:</strong> {segment.dominant_class}<br />
                      <strong>CLIP Prediction:</strong> {segment.clip_prediction ? 
                        `${segment.clip_prediction} (Confidence: ${(segment.clip_confidence * 100).toFixed(1)}%)` : 
                        "Not available"}<br />
                      <strong>From:</strong> {segment.start_timestamp} (Frame: {segment.start_frame_idx})<br />
                      <strong>To:</strong> {segment.end_timestamp} (Frame: {segment.end_frame_idx})<br />
                      <strong>Duration:</strong> {segment.duration_seconds} seconds<br />
                      <strong>Events in segment:</strong> {segment.event_count_in_segment}
                    </p>
                    
                    {segment.representative_frames && segment.representative_frames.length > 0 && (
                      <Box mt={2}>
                        <Typography variant="subtitle2" style={{ color: '#000', fontWeight: 'bold', fontFamily: 'inherit', fontSize: 'inherit' }}>
                          Representative Frames:
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'row', gap: '10px', flexWrap: 'wrap', mt: 1 }}>
                          {segment.representative_frames.map((frame, frameIndex) => (
                            <img
                              key={frameIndex}
                              src={`${API_URL}/inferred_frames/${frame.saved_frame_path}`}
                              
                              alt={`Representative frame ${frameIndex + 1} for segment ${index + 1}`}
                              crossOrigin="anonymous"  // <--- Add this
                              style={{ maxHeight: '400px', maxWidth: '400px', border: '1px solid #ddd', borderRadius: '4px' }}
                            />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {evidenceError && <p className="error-message form-error-message">Evidence Extraction Error: {evidenceError}</p>}
        {evidenceExtractionResult && (
          <div ref={evidenceResultsRef} className="evidence-results styled-box">
            <h3>== OBJECT DETECTION RESULTS ==</h3>
            <pre className="evidence-text">
              {evidenceExtractionResult.replace('🔍 Object Detection Results:\n🔍 Object Detection Results:', '🔍 Object Detection Results:')}
            </pre>
            
            {evidenceExtractionResult.includes('Frame with most detections') && (
              <div className="best-frame-container">
                <h4>Frame with Most Detections:</h4>
                {evidenceExtractionResult.split('\n').map((line, index) => {
                  if (line.includes('Frame saved as:')) {
                    const framePath = line.split('Frame saved as:')[1].trim();
                    const fullPath = framePath.replace(/\\/g, '/');
                    return (
                      <div key={index} className="best-frame-wrapper">
                        <img 
                          src={`http://localhost:5000/proxy_inferred_frame/${fullPath}`}

                          //src={`${API_URL}/inferred_frames/${fullPath}`} 
                          alt="Frame with most detections"
                          className="best-frame-image"
                          crossOrigin="anonymous"  // <--- Add this
                          onError={(e) => {
                            console.error("Best frame image failed to load:", e);
                            e.target.style.display = 'none';
                            const errorDiv = document.createElement('div');
                            errorDiv.className = 'frame-error';
                            errorDiv.textContent = 'Failed to load frame image';
                            e.target.parentNode.appendChild(errorDiv);
                          }}
                        />
                        <p className="frame-path">Path: {fullPath}</p>
                      </div>
                    );
                  }
                  return null;
                })}
              </div>
            )}

            {evidenceExtractionResult.includes('Suspicious Objects Detected:') && (
              <div className="suspicious-frames-container">
                <h4>⚠️ Frames with Suspicious Objects:</h4>
                <div className="suspicious-frames-grid">
                  {evidenceExtractionResult.split('\n').map((line, index) => {
                    if (line.includes('suspicious_frame_')) {
                      const framePath = line.trim().split('Frame saved as: ')[1];
                      const fullPath = framePath.replace(/\\/g, '/');
                      return (
                        <div key={index} className="suspicious-frame-item">
                          <img 
                            src={`${API_URL}/inferred_frames/${fullPath}`}
                            alt="Frame with suspicious objects"
                            className="suspicious-frame-image"
                            crossOrigin="anonymous"  // <--- Add this
                            onError={(e) => {
                              console.error("Suspicious frame image failed to load:", e);
                              e.target.style.display = 'none';
                              const errorDiv = document.createElement('div');
                              errorDiv.className = 'frame-error';
                              errorDiv.textContent = 'Failed to load frame image';
                              e.target.parentNode.appendChild(errorDiv);
                            }}
                          />
                          <p className="frame-path">Path: {fullPath}</p>
                        </div>
                      );
                    }
                    return null;
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {captionError && <p className="error-message form-error-message">Captioning Error: {captionError}</p>}
        {captionResult && captionResult.caption && (
          <div className="caption-results styled-box">
            <h3>== SYSTEM_NARRATIVE ==</h3>
            <div className="caption-text markdown-content">
              {displayedCaption.split('\n').map((line, index) => {
                const cleanLine = line
                  .replace(/\s+/g, ' ')
                  .replace(/([a-z])([A-Z])/g, '$1 $2')
                  .replace(/([.,!?])([A-Za-z])/g, '$1 $2')
                  .trim();

                if (cleanLine.startsWith('##')) {
                  return <h2 key={index} className="caption-header">{cleanLine.replace('##', '').trim()}</h2>;
                } else if (cleanLine.startsWith('-')) {
                  return <li key={index} className="caption-list-item">{cleanLine.replace('-', '').trim()}</li>;
                } else if (cleanLine.trim() === '') {
                  return <br key={index} />;
                } else if (cleanLine.startsWith('**')) {
                  const boldText = cleanLine.replace(/\*\*/g, '');
                  return <strong key={index}>{boldText}</strong>;
                } else {
                  return <p key={index} className="caption-paragraph">{cleanLine}</p>;
                }
              })}
            </div>
            <button 
              onClick={() => navigator.clipboard.writeText(captionResult.caption)}
              className="pixel-button copy-button"
              disabled={displayedCaption.length !== (captionResult?.caption?.length || 0)}
            >
              COPY_NARRATIVE
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoAnalysisPage;