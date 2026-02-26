import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { Link } from "react-router-dom"
import "./DoctorSignup.css";

const DoctorSignup = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    specialization: "",
  });

 const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/api/doctors/signup", formData);
      setAlertMessage('Signup successful! Now you can Login.');
      setTimeout(() => {
        setFullName('');
        setEmail('');
        setPassword('');
      }, 1000); // Clear fields after showing message
      navigate('/doctor-dashboard')
    } catch (error) {
      alert(error.response.data.message || "Signup failed. Try again.");
    }
  };

  return (
    <div className="signup-container">
      <h2>Doctor Signup</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          name="name"
          placeholder="Name"
          value={formData.name}
          onChange={handleChange}
          required
        />
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={formData.email}
          onChange={handleChange}
          required
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
          required
        />
        <input
          type="text"
          name="specialization"
          placeholder="Specialization"
          value={formData.specialization}
          onChange={handleChange}
          required
        />
        <button type="submit">Signup</button>
        <p>Already an user? <Link to="/doctor-login">Here</Link></p>
      </form>
    </div>
  );
};

export default DoctorSignup;
