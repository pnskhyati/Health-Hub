import React, { useState } from 'react';
import { Link } from "react-router-dom";
import "./Navbar.css";
import Alert from './E-Pharmacy/Products/Alert';
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import NutritionChatbot from './Chatbot/NutritionChatbot'; // Import the chatbot component

export default function Navbar() {
  const user = JSON.parse(localStorage.getItem('user'));
  const [showChatbot, setShowChatbot] = useState(false);
  const handleLogout = () => {
    localStorage.removeItem('user');
    <Alert message='Logged out successfully!'/>
    window.location.reload();
  };

  return (
    <>
      <header className="navbar">
        <div className="logo">HealthHub</div>
        <nav className="nav-links">
          <a href="/">Home</a>
          <a href="/disease-detection">Disease Detection</a>
          <Link to="/order-medicines">Order Medicine</Link>
          <Link to="/book">Consult a Doctor</Link>
          <Link to="/medicine-schedule">Medicine Schedule</Link>
          <a href="/medical-records">Medical Records</a>
        </nav>
        <div className="right-section">
          <div className="auth-links">
            {user ? (
              <>
                <span>Welcome, {user.fullName}</span>
                <button onClick={handleLogout} className="logout-btn">Logout</button>
              </>
            ) : (
              <>
                <Link to="/choice-page" className="login-btn">Login</Link>
                <Link to="/choice-page" className="signup-btn">Sign Up</Link>
              </>
            )}
          </div>
          <button 
            className="chatbot-toggle" 
            onClick={() => setShowChatbot(!showChatbot)}
            aria-label="Toggle chatbot"
          >
            {showChatbot ? <CloseIcon /> : <ChatIcon />}
          </button>
        </div>
      </header>

      {/* Render the chatbot when showChatbot is true */}
      {showChatbot && (
        <div className="chatbot-container">
          <NutritionChatbot />
        </div>
      )}
    </>
  );
}