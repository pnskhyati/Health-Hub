const express = require('express');
const router = express.Router();
const Doctor = require('../models/Doctors');
const RegisteredDoctor = require('../models/DoctorsSchema');
const Appointment = require('../models/AppointmentSchema'); // Assuming Appointments schema exists

// Get Doctor's Profile
router.get('/profile/:doctorId', async (req, res) => {
    const { doctorId } = req.params;
    try {
        const doctor = await Doctor.findById(doctorId);
        if (!doctor) {
            return res.status(404).json({ message: "Doctor not found" });
        }
         // Fetch the doctor's specialization from the RegisteredDoctor collection
         const DoctorSpecialization = await RegisteredDoctor.findById(doctorId);
         if (!DoctorSpecialization) {
             return res.status(404).json({ message: "Doctor specialization not found" });
         }
         res.status(200).json({
            doctor: {
                name: doctor.name,
                email: doctor.email,
                availability: doctor.availability,
            },
            specialization: DoctorSpecialization.speciality, // Send specialization separately
        });
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
});

router.get('/appointments/:doctorId', async (req, res) => {
    const { doctorId } = req.params;
    try {
        // Fetch appointments for the doctor and populate patient data
        const appointments = await Appointment.find({ doctorId }).populate('userId', 'name email');
        
        // Map over the appointments to send the required data as per the schema
        const formattedAppointments = appointments.map(appt => ({
            appointmentId: appt._id,
            patientName: appt.patientName,
            patientEmail: appt.patientEmail,
            appointmentDate: appt.appointmentDate,
            appointmentTime: appt.appointmentTime,
            doctorId: appt.doctorId,
            doctorName: appt.doctorName,
            PatientId: appt.userId
        }));

        res.status(200).json(formattedAppointments);
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
});

// Update Doctor's Availability
router.put('/availability/:doctorId', async (req, res) => {
    const { doctorId } = req.params;
    const { availability } = req.body;
    try {
        const doctor = await Doctor.findByIdAndUpdate(
            doctorId,
            { availability },
            { new: true }
        );
        res.status(200).json({ message: "Availability updated", doctor });
    } catch (error) {
        res.status(500).json({ message: "Server error", error: error.message });
    }
});

module.exports = router;
