import React, { useState, useEffect } from "react";
import axios from "axios";
import "./MedicalRecord.css";

const MedicalRecords = () => {
  const [formData, setFormData] = useState({
    patientName: "",
    dateOfBirth: "",
    medicalHistory: "",
    medications: "",
  });
  const [reports, setReports] = useState([]);
  const [records, setRecords] = useState([]);
  
  // Assuming the token is stored in localStorage after login
  const token = localStorage.getItem("token");

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e) => {
    setReports(e.target.files);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    // Ensure token exists
    if (!token) {
      alert("You need to be logged in to save records.");
      return;
    }
  
    const data = new FormData();
    Object.keys(formData).forEach((key) => data.append(key, formData[key]));
    Array.from(reports).forEach((file) => data.append("reports", file));
  
    try {
      // Make the POST request with token in header
      await axios.post("http://localhost:5000/api/records", data, {
        headers: {
          "Content-Type": "multipart/form-data",
          "Authorization": `Bearer ${token}`,
        },
      });
      alert("Medical record saved!");
      fetchRecords();
    } catch (error) {
      console.error("Error saving record:", error);
      alert("Failed to save record");
    }
  };
  

  const fetchRecords = async () => {
    try {
      // Include the token in the request headers for authorization
      const response = await axios.get("http://localhost:5000/api/records", {
        headers: {
          "Authorization": `Bearer ${token}`,
        },
      });
      setRecords(response.data);
    } catch (error) {
      console.error("Error fetching records", error);
    }
  };

  useEffect(() => {
    if (token) {
      fetchRecords();
    } else {
      // If there's no token, you can redirect to login page or show a message
      alert("Please log in to view your medical records.");
    }
  }, [token]);

  return (
    <div className="medical-records">
      <h1>Medical Records</h1>
      <form onSubmit={handleSubmit} className="medical-records-form">
        <input
          type="text"
          name="patientName"
          placeholder="Patient Name"
          value={formData.patientName}
          onChange={handleInputChange}
          className="medical-records-input"
          required
        />
        <input
          type="date"
          name="dateOfBirth"
          value={formData.dateOfBirth}
          onChange={handleInputChange}
          className="medical-records-input"
          required
        />
        <textarea
          name="medicalHistory"
          placeholder="Medical History"
          value={formData.medicalHistory}
          onChange={handleInputChange}
          className="medical-records-textarea"
        ></textarea>
        <textarea
          name="medications"
          placeholder="Medications"
          value={formData.medications}
          onChange={handleInputChange}
          className="medical-records-textarea"
        ></textarea>
        <input
          type="file"
          multiple
          onChange={handleFileChange}
          className="medical-records-input"
        />
        <button type="submit" className="medical-records-button">
          Save Record
        </button>
      </form>
      <h2>Saved Records</h2>
      <div className="medical-records-list">
        {records.map((record, index) => (
          <div key={index} className="medical-records-item">
            <h3>{record.patientName}</h3>
            <p>DOB: {record.dateOfBirth}</p>
            <p>History: {record.medicalHistory}</p>
            <p>Medications: {record.medications}</p>
            <h4>Reports:</h4>
            <ul>
              {record.reports.map((url, i) => (
                <li key={i}>
                  <a
                    href={`http://localhost:5000${url}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    View Report {i + 1}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MedicalRecords;
