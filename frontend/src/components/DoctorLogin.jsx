import React, { useState } from 'react';
import './Login.css'; 
import Alert from './E-Pharmacy/Products/Alert'; 
import { useNavigate } from 'react-router-dom';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [alertMessage, setAlertMessage] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (event) => {
    event.preventDefault();
    console.log("Login attempted with", email, password);
    try {
      const response = await fetch("http://localhost:5000/api/doctors/login", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await response.json();
      if (response.status === 200) {
        localStorage.setItem('Doctor', JSON.stringify(data.user));  
        localStorage.setItem('DoctorId', data.user._id);  // Store the user ID
        console.log("Doctor_id: ",localStorage.getItem('DoctorId') );
        

        setAlertMessage('Login successful!');
        setTimeout(() => {
          navigate(`/doctors-dashboard`);
          window.location.reload();
        }, 1000); // wait for alert message
      } else {
        setAlertMessage(data.message); // Show error message from backend
      }
    } catch (error) {
      console.log('Error:', error);
      setAlertMessage('Login failed. Please try again.');
    }
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h2>Login</h2>
        <form onSubmit={handleLogin}>
          <div className="input-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <button type="submit" className="login-btn">Login</button>
        </form>
      </div>
      {alertMessage && <Alert message={alertMessage} />}
    </div>
  );
}

export default Login;
