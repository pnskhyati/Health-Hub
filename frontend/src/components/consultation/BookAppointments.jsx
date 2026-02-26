import React, { useEffect, useState } from 'react';
import { Modal, Button, Form } from 'react-bootstrap';
import Navbar from '../Navbar';
import "./BookAppointments.css";
import axios from "axios";

const BookAppointments = () => {
  const [specialty, setSpecialty] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [doctors, setDoctors] = useState([]);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [appointments, setAppointments] = useState([]);
  const [showAppointmentsModal, setShowAppointmentsModal] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (specialty) {
      setIsLoading(true);
      axios
        .get(`http://localhost:5000/api/doctors?specialty=${specialty}`)
        .then((response) => {
          setDoctors(response.data);
          setIsLoading(false);
        })
        .catch((error) => {
          console.error("Error fetching doctors:", error);
          setIsLoading(false);
        });
    } else {
      setDoctors([]);
    }
  }, [specialty]);

  const handleSpecialtyChange = (e) => {
    setSpecialty(e.target.value);
  };

  const handleBookClick = (doctor) => {
    setSelectedDoctor(doctor);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setSelectedDoctor(null);
  };

  const handleAppointmentSubmit = (e) => {
    e.preventDefault();

    const userId = localStorage.getItem("userId");

    if (!userId) {
      alert("You must be logged in to book an appointment.");
      return;
    }

    const appointmentDetails = {
      patientName: e.target.formName.value,
      patientEmail: e.target.formEmail.value,
      appointmentDate: e.target.formDate.value,
      appointmentTime: e.target.formTime.value,
      doctorId: selectedDoctor._id,
      doctorName: selectedDoctor["Doctor's Name"],
      userId,
    };

    axios
      .post(`http://localhost:5000/api/appointments`, appointmentDetails)
      .then((response) => {
        console.log("Appointment booked successfully:", response.data);
        alert("Appointment booked successfully!");
        handleCloseModal();
      })
      .catch((error) => {
        console.error("Error booking appointment:", error);
        alert("Failed to book appointment.");
      });
  };

  const fetchAppointments = () => {
    const userId = localStorage.getItem("userId");

    if (!userId) {
      alert("You must be logged in to view appointments.");
      return;
    }

    setIsLoading(true);
    axios
      .get(`http://localhost:5000/api/fetch-appointments?userId=${userId}`)
      .then((response) => {
        setAppointments(response.data);
        setShowAppointmentsModal(true);
        setIsLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching appointments:", error);
        setIsLoading(false);
      });
  };

  const handleCloseAppointmentsModal = () => {
    setShowAppointmentsModal(false);
  };

  return (
    <div className='background'>
      <div className='navbar-appointments'>
        <Navbar />
      </div>
      <div className="container appointment">
        <h1>Book Appointments</h1>
        <Form.Group controlId="specialtySelect" className='form-group'>
          <Form.Label className='form-label'>Select Specialty</Form.Label>
          <Form.Control className='form-control' as="select" value={specialty} onChange={handleSpecialtyChange}>
            <option value="">Select...</option>
            <option value="Dentist">Dentist</option>
            <option value="Gynecologist">Gynecologist</option>
            <option value="Acupuncturist">Acupuncturist</option>
            <option value="Cardiologist">Cardiologist</option>
            <option value="Dermatologist">Dermatologist</option>
          </Form.Control>
        </Form.Group>
        
        <div className="doctors-list-container">
          {isLoading ? (
            <div className="loading-spinner">
              <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
            </div>
          ) : (
            specialty && (
              <div className="scrollable-doctors-list">
                {doctors.map((doctor, index) => (
                  <div key={index} className="card mb-3 doctor-card">
                    <div className="card-body">
                      <h5 className="card-title">{doctor["Doctor's Name"]}</h5>
                      <p className="card-text specialty">{doctor.speciality}</p>
                      <p className="card-text experience">Experience: {doctor.experience || '5+'} years</p>
                      <Button
                        variant="primary"
                        className="book-button"
                        onClick={() => handleBookClick(doctor)}
                      >
                        Book Appointment
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )
          )}
          {specialty && !isLoading && doctors.length === 0 && (
            <div className="no-doctors">
              <p>No doctors available for this specialty.</p>
            </div>
          )}
        </div>
        
        <div className="view-appointments-container">
          <Button className="view-appointments-btn" variant="secondary" onClick={fetchAppointments}>
            View My Appointments
          </Button>
        </div>

        {/* Book Appointment Modal */}
        <Modal show={showModal} onHide={handleCloseModal} centered>
          <Modal.Header closeButton>
            <Modal.Title>
              Book Appointment with Dr. {selectedDoctor?.["Doctor's Name"]}
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form onSubmit={handleAppointmentSubmit}>
              <Form.Group controlId="formName" className="mb-3">
                <Form.Label>Full Name</Form.Label>
                <Form.Control type="text" placeholder="Enter your full name" required />
              </Form.Group>
              <Form.Group controlId="formEmail" className="mb-3">
                <Form.Label>Email Address</Form.Label>
                <Form.Control type="email" placeholder="Enter your email" required />
              </Form.Group>
              <Form.Group controlId="formDate" className="mb-3">
                <Form.Label>Preferred Date</Form.Label>
                <Form.Control type="date" required />
              </Form.Group>
              <Form.Group controlId="formTime" className="mb-3">
                <Form.Label>Preferred Time</Form.Label>
                <Form.Control type="time" required />
              </Form.Group>
              <div className="modal-actions">
                <Button variant="secondary" onClick={handleCloseModal}>
                  Cancel
                </Button>
                <Button variant="primary" type="submit">
                  Confirm Appointment
                </Button>
              </div>
            </Form>
          </Modal.Body>
        </Modal>

        {/* Appointments List Modal */}
        <Modal show={showAppointmentsModal} onHide={handleCloseAppointmentsModal} centered scrollable>
          <Modal.Header closeButton>
            <Modal.Title>Your Appointments</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            {isLoading ? (
              <div className="loading-spinner">
                <div className="spinner-border text-primary" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
              </div>
            ) : appointments.length > 0 ? (
              <ul className="appointments-list">
                {appointments.map((appointment, index) => (
                  <li key={index} className="appointment-item">
                    <div className="appointment-details">
                      <h5>Dr. {appointment.doctorName}</h5>
                      <p><strong>Date:</strong> {new Date(appointment.appointmentDate).toLocaleDateString()}</p>
                      <p><strong>Time:</strong> {appointment.appointmentTime}</p>
                      <p><strong>Status:</strong> <span className="status-confirmed">Confirmed</span></p>
                    </div>
                    <Button
                      variant="primary"
                      className="join-button"
                      onClick={() => window.location.href = `/room/${appointment.appointmentId}`}
                    >
                      Join Video Call
                    </Button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="no-appointments">
                <p>No appointments found.</p>
              </div>
            )}
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={handleCloseAppointmentsModal}>
              Close
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    </div>
  );
};

export default BookAppointments;