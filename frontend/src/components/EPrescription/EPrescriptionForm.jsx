import React, { useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./EPrescriptionForm.css";

const EPrescriptionForm = ({ doctorId, patientId, onPrescriptionCreated }) => {
  const [formData, setFormData] = useState({
    medicines: [{ name: "", dosage: "", duration: "", instructions: "" }],
    tests: [{ name: "", reason: "", instructions: "" }],
    notes: "",
    followUpDate: ""
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (type, index, field, value) => {
    const updatedItems = [...formData[type]];
    updatedItems[index][field] = value;
    setFormData({ ...formData, [type]: updatedItems });
  };

  const addItem = (type) => {
    const template = type === "medicines" 
      ? { name: "", dosage: "", duration: "", instructions: "" }
      : { name: "", reason: "", instructions: "" };
    
    setFormData({
      ...formData,
      [type]: [...formData[type], template]
    });
  };

  const removeItem = (type, index) => {
    const updatedItems = formData[type].filter((_, i) => i !== index);
    setFormData({ ...formData, [type]: updatedItems });
  };

  const validateForm = () => {
    // Validate medicines
    for (const med of formData.medicines) {
      if (!med.name || !med.dosage || !med.duration) {
        toast.error("Please fill all required fields for medicines");
        return false;
      }
    }

    // Validate tests
    for (const test of formData.tests) {
      if (!test.name) {
        toast.error("Please fill test name for all tests");
        return false;
      }
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setIsSubmitting(true);

    try {
      const response = await axios.post("http://localhost:5000/api/eprescriptions", {
        doctorId,
        patientId,
        medicines: formData.medicines,
        tests: formData.tests,
        notes: formData.notes,
        followUpDate: formData.followUpDate
      });

      toast.success("ePrescription created successfully!");
      
      // Reset form
      setFormData({
        medicines: [{ name: "", dosage: "", duration: "", instructions: "" }],
        tests: [{ name: "", reason: "", instructions: "" }],
        notes: "",
        followUpDate: ""
      });

      if (onPrescriptionCreated) {
        onPrescriptionCreated(response.data.savedPrescription);
      }
    } catch (error) {
      console.error("Error creating ePrescription:", error);
      toast.error(error.response?.data?.message || "Error creating ePrescription");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form className="eprescription-form" onSubmit={handleSubmit}>
      <h2>Create ePrescription</h2>

      <div className="form-section">
        <h3>Medications</h3>
        {formData.medicines.map((medicine, index) => (
          <div key={index} className="medicine-card">
            <div className="form-row">
              <div className="form-group">
                <label>Medicine Name*</label>
                <input
                  type="text"
                  value={medicine.name}
                  onChange={(e) => handleChange("medicines", index, "name", e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label>Dosage*</label>
                <input
                  type="text"
                  value={medicine.dosage}
                  onChange={(e) => handleChange("medicines", index, "dosage", e.target.value)}
                  placeholder="e.g., 1 tablet twice daily"
                  required
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Duration*</label>
                <input
                  type="text"
                  value={medicine.duration}
                  onChange={(e) => handleChange("medicines", index, "duration", e.target.value)}
                  placeholder="e.g., 7 days"
                  required
                />
              </div>
              <div className="form-group">
                <label>Instructions</label>
                <input
                  type="text"
                  value={medicine.instructions}
                  onChange={(e) => handleChange("medicines", index, "instructions", e.target.value)}
                  placeholder="e.g., Take after food"
                />
              </div>
            </div>

            {formData.medicines.length > 1 && (
              <button 
                type="button" 
                className="remove-btn"
                onClick={() => removeItem("medicines", index)}
              >
                Remove
              </button>
            )}
          </div>
        ))}
        <button 
          type="button" 
          className="add-btn"
          onClick={() => addItem("medicines")}
        >
          + Add Another Medicine
        </button>
      </div>

      <div className="form-section">
        <h3>Diagnostic Tests</h3>
        {formData.tests.map((test, index) => (
          <div key={index} className="test-card">
            <div className="form-row">
              <div className="form-group">
                <label>Test Name*</label>
                <input
                  type="text"
                  value={test.name}
                  onChange={(e) => handleChange("tests", index, "name", e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label>Reason</label>
                <input
                  type="text"
                  value={test.reason}
                  onChange={(e) => handleChange("tests", index, "reason", e.target.value)}
                  placeholder="Reason for test"
                />
              </div>
            </div>

            <div className="form-group">
              <label>Instructions</label>
              <input
                type="text"
                value={test.instructions}
                onChange={(e) => handleChange("tests", index, "instructions", e.target.value)}
                placeholder="Any special instructions"
              />
            </div>

            {formData.tests.length > 1 && (
              <button 
                type="button" 
                className="remove-btn"
                onClick={() => removeItem("tests", index)}
              >
                Remove
              </button>
            )}
          </div>
        ))}
        <button 
          type="button" 
          className="add-btn"
          onClick={() => addItem("tests")}
        >
          + Add Another Test
        </button>
      </div>

      <div className="form-section">
        <h3>Additional Information</h3>
        <div className="form-group">
          <label>Follow-up Date</label>
          <input
            type="date"
            value={formData.followUpDate}
            onChange={(e) => setFormData({ ...formData, followUpDate: e.target.value })}
          />
        </div>
        
        <div className="form-group">
          <label>Doctor's Notes</label>
          <textarea
            value={formData.notes}
            onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
            placeholder="Additional notes for the patient"
            rows="4"
          ></textarea>
        </div>
      </div>

      <div className="form-actions">
        <button 
          type="submit" 
          className="submit-btn"
          disabled={isSubmitting}
        >
          {isSubmitting ? "Saving..." : "Save Prescription"}
        </button>
      </div>
    </form>
  );
};

export default EPrescriptionForm;