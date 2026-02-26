import React, { useEffect, useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";
import EPrescriptionForm from "./EPrescription/EPrescriptionForm"; // Import your prescription form
import "./DoctorsDashboard.css";

const DoctorDashboard = () => {
    const [doctor, setDoctor] = useState(null);
    const [specialization, setSpecialization] = useState("");
    const [appointments, setAppointments] = useState([]);
    const [availability, setAvailability] = useState("");
    const [selectedAppointment, setSelectedAppointment] = useState(null); // Track selected appointment
    const [showPrescriptionModal, setShowPrescriptionModal] = useState(false); // Control modal visibility

    const doctorId = localStorage.getItem("DoctorId");

    useEffect(() => {
        const fetchDoctorProfile = async () => {
            try {
                const response = await axios.get(`http://localhost:5000/api/dashboard/profile/${doctorId}`);
                setDoctor(response.data.doctor);
                setSpecialization(response.data.specialization);
            } catch (error) {
                console.error("Error fetching doctor profile:", error);
            }
        };

        fetchDoctorProfile();
    }, [doctorId]);

    useEffect(() => {
        const fetchAppointments = async () => {
            try {
                const response = await axios.get(`http://localhost:5000/api/dashboard/appointments/${doctorId}`);
                setAppointments(response.data);
            } catch (error) {
                console.error("Error fetching appointments:", error);
            }
        };

        fetchAppointments();
    }, [doctorId]);

    const handleAvailabilityUpdate = async () => {
        try {
            const response = await axios.put(`http://localhost:5000/api/dashboard/availability/${doctorId}`, {
                availability,
            });
            alert(response.data.message);
            setDoctor((prev) => ({ ...prev, availability }));
        } catch (error) {
            console.error("Error updating availability:", error);
        }
    };

    return (
        <div className="doctor-dashboard">
            <h1>Doctor's Dashboard</h1>

            {doctor && (
                <div className="profile">
                    <h2>Welcome, Dr. {doctor.name}</h2>
                    <p>Email: {doctor.email}</p>
                    <p>Specialization: {specialization}</p>
                </div>
            )}

            <div className="availability">
                <h3>Set Your Availability</h3>
                <input
                    type="text"
                    placeholder="e.g., Mon-Fri 9 AM - 5 PM"
                    value={availability}
                    onChange={(e) => setAvailability(e.target.value)}
                />
                <button onClick={handleAvailabilityUpdate}>Update</button>
            </div>

            <div className="appointments">
                <h3>Upcoming Appointments</h3>
                {appointments.length > 0 ? (
                    <ul>
                    {appointments.length > 0 ? (
                        appointments.map((appt) => (
                            <li key={appt.appointmentId}> {/* Use appointmentId here */}
                                <p>Patient: {appt.patientName}</p>
                                <p>Email: {appt.patientEmail}</p>
                                <p>Date: {new Date(appt.appointmentDate).toLocaleDateString()}</p>
                                <p>Time: {appt.appointmentTime}</p>
                
                                <Link to={`/room/${appt.appointmentId}`}>
                                    <button>Start Meeting</button>
                                </Link>
                
                                <button
                                    onClick={() => {
                                        setSelectedAppointment(appt);
                                        setShowPrescriptionModal(true); // Show modal
                                    }}
                                >
                                    Generate ePrescription
                                </button>
                            </li>
                        ))
                    ) : (
                        <p>No appointments scheduled.</p>
                    )}
                </ul>
                
                ) : (
                    <p>No appointments scheduled.</p>
                )}
            </div>

            {/* Modal for ePrescription Form */}
            <div className={`prescription-modal ${showPrescriptionModal ? "active" : ""}`}>
                <div className="modal-content">
                    {selectedAppointment && (
                        <EPrescriptionForm
                            doctorId={doctorId}
                            patientId={selectedAppointment.PatientId}
                            appointmentId={selectedAppointment._id}
                        />
                    )}
                    <button
                        onClick={() => setShowPrescriptionModal(false)} // Close modal
                        className="close-btn"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default DoctorDashboard;
