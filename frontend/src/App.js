import React from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  Navigate,
  Outlet
} from 'react-router-dom';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import VideoAnalysisPage from './pages/VideoAnalysisPage';
import { useAuth } from './contexts/AuthContext';

import './App.css';
// import logoImage from './assets/logo.png'; // Example: Uncomment when you have a logo

// ProtectedRoute component
const ProtectedRoute = () => {
  const { currentUser, loading } = useAuth();

  if (loading) {
    return <div className="global-loading">Loading application...</div>;
  }

  return currentUser ? <Outlet /> : <Navigate to="/login" replace />;
};

// Layout component for consistent navigation
const Layout = () => {
  const { currentUser, logout, loading } = useAuth();

  if (loading && !currentUser) {
    return <div className="global-loading">Loading SceneSolver...</div>
  }

  return (
    <div className="app-layout">
      <nav className="navbar">
        <div className="navbar-logo">
          {/* <img src={logoImage} alt="SceneSolver Logo" style={{ height: '40px', marginRight: '10px' }} /> */}
          <Link to="/">🔎</Link> {/* Placeholder text, can be replaced/enhanced by logo */}
        </div>
        <div className="navbar-links">
          <ul>
            {/* We can add a Home link that shows for everyone if desired, or keep it conditional */}
            {/* For simplicity, keeping Home link in logo/app name for now */}
            {currentUser ? (
              <>
                <li><Link to="/video-analysis">Video Analysis</Link></li>
                <li><button onClick={logout}>Logout</button></li>
                {/* Optionally display username: <li style={{color: 'white'}}>Welcome, {currentUser.username}</li> */}
              </>
            ) : (
              <>
                <li><Link to="/">Home</Link></li> {/* Explicit Home for logged out users */}
                <li><Link to="/login">Login</Link></li>
                <li><Link to="/signup">Sign Up</Link></li>
              </>
            )}
          </ul>
        </div>
      </nav>
      <main className="main-content">
        <Outlet /> {/* Nested routes will render here */}
      </main>
    </div>
  );
};

function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/video-analysis" element={<VideoAnalysisPage />} />
          </Route>
          <Route path="*" element={<div><h2>404 - Page Not Found</h2><Link to="/">Go to Home</Link></div>} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
