import React, { createContext, useState, useEffect, useContext } from 'react';
import * as authService from '../services/authService';

const AuthContext = createContext();

export const useAuth = () => {
  return useContext(AuthContext);
};

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkLoggedIn = async () => {
      setLoading(true);
      try {
        const response = await authService.getCurrentUser();
        if (response.success && response.user) {
          setCurrentUser(response.user);
        } else {
          setCurrentUser(null);
        }
      } catch (error) {
        // Error already handled in authService to return { success: false, ... }
        // console.error("Error checking login status:", error);
        setCurrentUser(null);
      }
      setLoading(false);
    };
    checkLoggedIn();
  }, []);

  const login = async (username, password) => {
    setLoading(true);
    try {
      const response = await authService.login(username, password);
      if (response.success && response.user) {
        setCurrentUser(response.user);
        setLoading(false);
        return response; // Return full response for component to handle messages/navigation
      } else {
        setCurrentUser(null);
        setLoading(false);
        return response; // Return error response
      }
    } catch (error) {
      setCurrentUser(null);
      setLoading(false);
      // Axios error might not have response.data if network error, etc.
      return error.response ? error.response.data : { success: false, message: 'Login failed. Please try again.' };
    }
  };

  const signup = async (username, email, password, confirm_password) => {
    setLoading(true);
    try {
      const response = await authService.signup(username, email, password, confirm_password);
      setLoading(false);
      return response; // Component will handle success/error message and navigation
    } catch (error) {
      setLoading(false);
      return error.response ? error.response.data : { success: false, message: 'Signup failed. Please try again.' };
    }
  };

  const logout = async () => {
    setLoading(true);
    try {
      await authService.logout();
      setCurrentUser(null);
    } catch (error) {
      console.error("Logout failed:", error);
      // Even if logout API call fails, clear user from frontend as a fallback
      setCurrentUser(null);
    }
    setLoading(false);
  };

  const value = {
    currentUser,
    loading,
    login,
    signup,
    logout,
    // setCurrentUser might be useful for direct manipulation in some edge cases, but usually via auth functions
  };

  return (
    <AuthContext.Provider value={value}>
      {children} {/* No need for !loading here, App can decide based on context loading state */}
    </AuthContext.Provider>
  );
};

// export default AuthContext; // Default export is not typically the context itself, but the provider or custom hook 