import React, { useState } from 'react';
import './Signup.css'; 
import Alert from './E-Pharmacy/Products/Alert';

function Signup() {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [alertMessage, setAlertMessage] = useState('');

  const handleSignup = async (event) => {
    event.preventDefault();
    console.log("Signup attempted with", fullName, email, password);
    try {
      const response = await fetch("http://localhost:5000/api/patients/signup", {
        method: "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ fullName, email, password })
      });

      const data = await response.json();

      if (response.status === 201) {
        setAlertMessage('Signup successful! Now you can Login.');
        setTimeout(() => {
          setFullName('');
          setEmail('');
          setPassword('');
        }, 1000); // Clear fields after showing message
      } else if (response.status === 409) {
        setAlertMessage('User already exists. Please login.');
      } else {
        setAlertMessage('An error occurred. Please try again.');
      }

    } catch (error) {
      console.error('Error:', error);
      setAlertMessage('An error occurred. Please try again.');
    }
  };

  return (
    <div className="signup-container">
      <div className="signup-box">
        <h2>Sign Up</h2>
        <form onSubmit={handleSignup}>
          <div className="input-group">
            <label htmlFor="fullName">Full Name</label>
            <input
              type="text"
              id="fullName"
              placeholder="Enter your full name"
              value={fullName}
              autoComplete="name"
              onChange={(e) => setFullName(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
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
              autoComplete="new-password" 
            />
          </div>
          <button type="submit" className="signup-btn">Sign Up</button>
        </form>
      </div>
      {alertMessage && <Alert message={alertMessage} />}
    </div>
  );
}

export default Signup;
