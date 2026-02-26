import React from "react";
import { useNavigate } from "react-router-dom";
import "./ChoicePage.css";

const ChoicePage = () => {
  const navigate = useNavigate();

  const handleChoice = (role) => {
    if (role === "doctor") {
      navigate("/doctor-signup");
    } else {
      navigate("/login-patient");
    }
  };

  return (
    <div className="choice-container">
      <h1>Welcome to HealthHub</h1>
      <p>Are you a Doctor or a Patient?</p>
      <div className="choice-buttons">
        <button onClick={() => handleChoice("doctor")}>Doctor</button>
        <button onClick={() => handleChoice("patient")}>Patient</button>
      </div>
    </div>
  );
};

export default ChoicePage;
