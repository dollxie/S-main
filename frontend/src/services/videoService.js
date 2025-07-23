// This file will contain functions for video processing API calls (upload, caption, progress)
import axios from 'axios';

//const API_URL = process.env.REACT_APP_API_URL || 'https://scene-solver.onrender.com/api'; // Use environment variable or default to Render URL

const API_URL =`${process.env.REACT_APP_API_URL}/api` || 'https://3ed0-103-88-237-251.ngrok-free.app/api'; // Use environment variable or default to localhost

export const uploadVideoForClassification = async (formData) => {
  // formData should contain the video file under the key 'video'
  const response = await axios.post(`${API_URL}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    withCredentials: true
  });
  return response.data; // Expects { success: true, data: { possible_scenes: [], text_color: '' } } or error
};

export const uploadVideoForCaptioning = async (formData) => {
  // formData should contain the video file under the key 'video'
  const response = await axios.post(`${API_URL}/caption`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    withCredentials: true
  });
  return response.data; // Expects { success: true, data: { caption: '' } } or error
};

export const getCaptionProgress = async () => {
  // This route might not be fully utilized yet based on current backend
  try {
    const response = await axios.get(`${API_URL}/caption_progress`);
    return response.data; // Expects { status: '', progress: 0, message: '' }
  } catch (error) {
    console.error("Error fetching caption progress:", error);
    return { status: 'error', progress: 0, message: 'Could not fetch progress.' };
  }
};

export const extractEvidence = async (formData) => {
  const response = await axios.post(`${API_URL}/extract-evidence`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    withCredentials: true
  });
  return response.data;
};

// Remove or comment out unused functions
// const sampleFrames = async (videoFile) => {
//   // ... existing code ...
// };

// const runYOLODetection = async (videoFile) => {
//   // ... existing code ...
// };

// const passToGemini = async (videoFile) => {
//   // ... existing code ...
// }; 