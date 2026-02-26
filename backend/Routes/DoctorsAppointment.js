const express = require("express");
const router = express.Router();
const Doctor = require("../models/DoctorsSchema");
const Appointment = require("../models/AppointmentSchema");
const authenticateUser = require("../middleware/authenticationToken");
const moment = require("moment");

// Fetch doctors by specialty
router.get("/doctors", async (req, res) => {
    try {
      const { specialty } = req.query;
      const doctors = await Doctor.find({ speciality: specialty });
      res.json(doctors);
    } catch (error) {
      res.status(500).json({ error: "Error fetching doctors." });
    }
  });


  // Save appointment details
router.post("/appointments", async (req, res) => {


  try {
    console.log("Inside appointments");
    const { userId,  appointmentDate, appointmentTime } = req.body; // Get userId from the request body (from frontend)
    
    if (!userId) {
      return res.status(400).json({ error: "User ID is required." });
    }

    if ( !appointmentDate || !appointmentTime) {
      return res.status(400).json({ error: "All fields are required." });
    }

        // Calculate notification time (15 minutes before appointment)
    const appointmentDateTime = moment(`${appointmentDate} ${appointmentTime}`, "YYYY-MM-DD HH:mm");
    const notificationTime = appointmentDateTime.subtract(15, "minutes").toDate();
    

    const appointment = new Appointment({ ...req.body, userId,  notificationTime  });
    await appointment.save();
    res.status(201).json({ message: "Appointment booked successfully!" });
  } catch (error) {
    console.error("Error booking appointment:", error);
    res.status(500).json({ error: "Error booking appointment." });
  }
});

// Fetch appointments for logged-in user

router.get("/fetch-appointments", async (req, res) => {
  try {
    const { userId } = req.query; // Get userId from the query parameters
    if (!userId) {
      return res.status(400).json({ error: "User ID is required." });
    }

    const appointments = await Appointment.find({ userId });
    res.json(appointments);
  } catch (error) {
    console.error("Error fetching appointments:", error);
    res.status(500).json({ error: "Error fetching appointments." });
  }
});


module.exports = router;