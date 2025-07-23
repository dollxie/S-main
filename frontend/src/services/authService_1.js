// This file will contain functions for authentication-related API calls (login, signup, logout, me)
import axios from 'axios';

const API_URL =`${process.env.REACT_APP_API_URL}/api` || 'https://3ed0-103-88-237-251.ngrok-free.app/api'; // Proxied by react-scripts dev server

export const login = async (username, password) => {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('password', password);

  const response = await axios.post(`${API_URL}/login`, params, {
    headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
    },
    withCredentials: true
  });
  return response.data; // Should contain { success: true, user: {...}, message: ... } or error
};

export const signup = async (username, email, password, confirm_password) => {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('email', email);
  params.append('password', password);
  params.append('confirm_password', confirm_password);

  const response = await axios.post(`${API_URL}/register`, params, {
    headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
    },
    withCredentials: true
  });
  return response.data; // Should contain { success: true, message: ... } or error
};

export const logout = async () => {
  const response = await axios.get(`${API_URL}/logout`, {
    withCredentials: true
  });
  return response.data; // Should contain { success: true, message: ... }
};

export const getCurrentUser = async () => {
  try {
    const response = await axios.get(`${API_URL}/me`, {
      withCredentials: true
    });
    return response.data; // Should contain { success: true, user: {...} } or an error if not authenticated
  } catch (error) {
    // If the /me route returns 401, axios will throw an error.
    // We can catch this and return a specific structure or let it propagate.
    if (error.response && error.response.data) {
      return error.response.data; // The backend already sends { success: false, message: ... }
    }
    // Fallback for unexpected errors
    return { success: false, message: 'Could not fetch user status.' };
  }
}; 