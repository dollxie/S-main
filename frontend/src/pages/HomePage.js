import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';

function HomePage() {
  return (
    <div className="home-page">
      <header className="hero-section">
        <h1>SceneSolver</h1>
        <p className="tagline">Advanced Video Analysis & Crime Scene Investigation Platform</p>
      </header>

      <section className="features-section">
        <h2>Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3>Scene Classification</h3>
            <p>Advanced AI-powered analysis of video content to identify potential criminal activities and suspicious behaviors.</p>
          </div>
          <div className="feature-card">
            <h3>Object Detection</h3>
            <p>Real-time detection of suspicious objects and weapons using state-of-the-art YOLOv8 model.</p>
          </div>
          <div className="feature-card">
            <h3>Narrative Generation</h3>
            <p>AI-generated detailed descriptions of video content using Google's Gemini model.</p>
          </div>
        </div>
      </section>

      <section className="models-section">
        <h2>AI Models Used</h2>
        <div className="models-grid">
          <div className="model-card">
            <h3>CLIP (Contrastive Language-Image Pre-training)</h3>
            <p>Used for initial frame-by-frame analysis to identify potential crime scenes and suspicious activities.</p>
          </div>
          <div className="model-card">
            <h3>Timesformer</h3>
            <p>Advanced video understanding model for temporal analysis and event classification.</p>
          </div>
          <div className="model-card">
            <h3>YOLOv8</h3>
            <p>State-of-the-art object detection model for identifying weapons and suspicious objects.</p>
          </div>
          <div className="model-card">
            <h3>Gemini</h3>
            <p>Google's multimodal AI model for generating detailed video descriptions and narratives.</p>
          </div>
        </div>
      </section>

      <section className="documentation-section">
        <h2>Documentation</h2>
        <div className="docs-grid">
          <div className="doc-card">
            <h3>Getting Started</h3>
            <ul>
              <li>Upload video files in MP4, AVI, MOV, or MKV format</li>
              <li>Maximum file size: 100MB</li>
              <li>Supported resolutions: Up to 4K</li>
            </ul>
          </div>
          <div className="doc-card">
            <h3>Analysis Features</h3>
            <ul>
              <li>Scene Classification</li>
              <li>Object Detection</li>
              <li>Narrative Generation</li>
              <li>Evidence Extraction</li>
            </ul>
          </div>
          <div className="doc-card">
            <h3>Output Formats</h3>
            <ul>
              <li>Detailed classification reports</li>
              <li>Object detection results with bounding boxes</li>
              <li>AI-generated narratives</li>
              <li>Timeline of suspicious events</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="cta-section">
        <h2>Ready to Get Started?</h2>
        <div className="cta-buttons">
          <Link to="/login" className="cta-button primary">Login</Link>
          <Link to="/signup" className="cta-button secondary">Sign Up</Link>
        </div>
      </section>
    </div>
  );
}

export default HomePage; 