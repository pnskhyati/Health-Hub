import React, { useState, useEffect } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./MedicineSchedule.css";
import { v4 as uuidv4 } from 'uuid';

export default function MedicineSchedule() {
  const [medicines, setMedicines] = useState([]);
  const [takenMedicines, setTakenMedicines] = useState({});
  const [newMedicine, setNewMedicine] = useState({ 
    name: "", 
    dosage: "", 
    duration: "", 
    instructions: "",
    time: ""
  });
  const [prescriptions, setPrescriptions] = useState([]);
  const [showPrescriptions, setShowPrescriptions] = useState(false);
  const [autoSchedule, setAutoSchedule] = useState(false);

  const patientId = localStorage.getItem("userId");

  useEffect(() => {
    const savedTakenMedicines = JSON.parse(localStorage.getItem("takenMedicines")) || {};
    setTakenMedicines(savedTakenMedicines);
  }, []);

  useEffect(() => {
    localStorage.setItem("takenMedicines", JSON.stringify(takenMedicines));
  }, [takenMedicines]);

  const handleAddMedicine = () => {
    if (newMedicine.name && newMedicine.dosage && newMedicine.duration) {
      const newMedicineWithId = {
        ...newMedicine,
        id: uuidv4(),
        day: 1 // Default to day 1 when manually added
      };
      setMedicines([...medicines, newMedicineWithId]);
      setNewMedicine({ name: "", dosage: "", duration: "", instructions: "", time: "" });
      toast.success(`${newMedicine.name} added to schedule!`);
    } else {
      toast.warning("Please fill in all required fields (Name, Dosage, Duration)");
    }
  };

  const fetchPrescriptions = async () => {
    if (!patientId) {
      toast.error("No patient ID found. Please log in again.");
      return;
    }

    try {
      const response = await fetch(
        `http://localhost:5000/api/fetch/eprescriptions?patientId=${patientId}`
      );
      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Failed to fetch prescriptions");

      setPrescriptions(data);
      setShowPrescriptions(true);
      toast.success("Prescriptions loaded successfully!");
    } catch (error) {
      console.error("Error fetching prescriptions:", error);
      toast.error(error.message || "Failed to load prescriptions");
    }
  };

  const generateAutoSchedule = () => {
    if (prescriptions.length === 0) {
      toast.warning("No prescriptions found to generate a schedule.");
      return;
    }

    let generatedSchedule = [];

    prescriptions.forEach((prescription) => {
      prescription.medicines.forEach((medicine) => {
        const dosage = medicine.dosage.toLowerCase();
        let times = [];

        if (dosage.includes("twice")) {
          times = ["08:00", "20:00"]; // Morning and evening
        } else if (dosage.includes("thrice")) {
          times = ["08:00", "14:00", "20:00"]; // Morning, afternoon, evening
        } else if (dosage.includes("once")) {
          times = ["08:00"]; // Default to morning
        } else {
          // Handle other cases or default to once daily
          times = ["08:00"];
        }

        const duration = parseInt(medicine.duration.match(/\d+/) || 1, 10);

        for (let day = 1; day <= duration; day++) {
          times.forEach((time) => {
            generatedSchedule.push({
              id: uuidv4(),
              name: medicine.name,
              dosage: medicine.dosage,
              time: time,
              day: day,
              duration: duration,
              instructions: medicine.instructions || "Take as directed",
            });
          });
        }
      });
    });

    // Sort by day and then by time
    generatedSchedule.sort((a, b) => {
      if (a.day !== b.day) return a.day - b.day;
      return a.time.localeCompare(b.time);
    });

    setMedicines(generatedSchedule);
    setAutoSchedule(true);
    toast.success(`Automated schedule created for ${generatedSchedule.length} doses!`);
  };

  useEffect(() => {
    const interval = setInterval(() => {
      const currentTime = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false
      }).replace(/^(\d{1,2}):(\d{2})/, (match, h, m) => 
        `${h.padStart(2, '0')}:${m}`
      );

      medicines.forEach((medicine) => {
        if (medicine.time === currentTime && !takenMedicines[medicine.id]) {
          toast.info(
            <div>
              <strong>Reminder: Day {medicine.day}</strong><br />
              Time to take {medicine.name} at {medicine.time}<br />
              {medicine.instructions && <small>{medicine.instructions}</small>}
            </div>,
            { autoClose: 15000 }
          );
        }
      });
    }, 60000);

    return () => clearInterval(interval);
  }, [medicines, takenMedicines]);

  const handleCheckMedicine = (medicineId) => {
    setTakenMedicines((prev) => ({
      ...prev,
      [medicineId]: !prev[medicineId],
    }));
  };

  // Group medicines by day
  const medicinesByDay = medicines.reduce((acc, medicine) => {
    const day = medicine.day || 1;
    if (!acc[day]) acc[day] = [];
    acc[day].push(medicine);
    return acc;
  }, {});

  return (
    <div className="medicine-schedule">
      <h1>Medicine Schedule</h1>

      <div className="schedule-form">
        <h2>Add New Medicine</h2>
        <div className="form-grid">
          <input
            type="text"
            placeholder="Medicine Name*"
            value={newMedicine.name}
            onChange={(e) => setNewMedicine({ ...newMedicine, name: e.target.value })}
          />
          <input
            type="text"
            placeholder="Dosage* (e.g., Twice a day)"
            value={newMedicine.dosage}
            onChange={(e) => setNewMedicine({ ...newMedicine, dosage: e.target.value })}
          />
          <input
            type="text"
            placeholder="Duration* (e.g., 5 days)"
            value={newMedicine.duration}
            onChange={(e) => setNewMedicine({ ...newMedicine, duration: e.target.value })}
          />
          <input
            type="text"
            placeholder="Time (e.g., 08:00)"
            value={newMedicine.time}
            onChange={(e) => setNewMedicine({ ...newMedicine, time: e.target.value })}
          />
          <input
            type="text"
            placeholder="Instructions (e.g., After food)"
            value={newMedicine.instructions}
            onChange={(e) => setNewMedicine({ ...newMedicine, instructions: e.target.value })}
          />
        </div>
        <button onClick={handleAddMedicine}>Add Medicine</button>
      </div>

      <div className="action-buttons">
        <button onClick={fetchPrescriptions} className="view-prescriptions-btn">
          {showPrescriptions ? "Hide Prescriptions" : "View Prescriptions"}
        </button>
       
      </div>

      <div className="schedule-list">
        <h2>Your Medication Plan</h2>
        {medicines.length === 0 ? (
          <div className="empty-state">
            <p>No medicines scheduled yet.</p>
            <p>Add medicines manually or generate from prescriptions.</p>
          </div>
        ) : (
          Object.entries(medicinesByDay).map(([day, dayMedicines]) => (
            <div key={`day-${day}`} className="day-schedule">
              <h3>Day {day}</h3>
              <ul>
                {dayMedicines.map((medicine) => (
                  <li key={medicine.id} className={takenMedicines[medicine.id] ? "checked" : ""}>
                    <input
                      type="checkbox"
                      checked={!!takenMedicines[medicine.id]}
                      onChange={() => handleCheckMedicine(medicine.id)}
                    />
                    <div className="medicine-details">
                      <span className="medicine-name">{medicine.name}</span>
                      <span className="medicine-time">{medicine.time}</span>
                      <span className="medicine-dosage">{medicine.dosage}</span>
                      {medicine.instructions && (
                        <span className="medicine-instructions">{medicine.instructions}</span>
                      )}
                      {takenMedicines[medicine.id] && (
                        <span className="taken-badge">Taken</span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))
        )}
      </div>

      {showPrescriptions && (
        <div className="prescriptions-list">
          <h2>Your Prescriptions</h2>
          {prescriptions.length === 0 ? (
            <p>No prescriptions found.</p>
          ) : (
            <div className="prescription-cards">
              {prescriptions.map((prescription, index) => (
                <div key={index} className="prescription-item">
                  <div className="prescription-header">
                    <h3>Prescription #{index + 1}</h3>
                    <span>
                    {!autoSchedule && (
                    <button onClick={generateAutoSchedule} className="auto-schedule-btn">
                      Auto-Generate Schedule
                  </button>
                    )}
                    </span>
                    <span className="prescription-date">
                      {new Date(prescription.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  
                  <div className="prescription-section">
                    <h4>Medicines</h4>
                    {prescription.medicines.length > 0 ? (
                      <ul>
                        {prescription.medicines.map((med, medIndex) => (
                          <li key={medIndex}>
                            <strong>{med.name}</strong> - {med.dosage} for {med.duration}
                            {med.instructions && <div className="instructions">{med.instructions}</div>}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p>No medicines prescribed.</p>
                    )}
                  </div>

                  {prescription.tests.length > 0 && (
                    <div className="prescription-section">
                      <h4>Tests</h4>
                      <ul>
                        {prescription.tests.map((test, testIndex) => (
                          <li key={testIndex}>
                            <strong>{test.name}</strong> - {test.reason}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {prescription.notes && (
                    <div className="prescription-section">
                      <h4>Doctor's Notes</h4>
                      <p>{prescription.notes}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <ToastContainer 
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={true}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />
    </div>
  );
}