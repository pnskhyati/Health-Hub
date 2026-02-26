const express = require('express');
const router = express.Router();
const EPrescription = require('../models/ePrescription');
const Appointment = require('../models/AppointmentSchema'); // Assuming Appointment model is defined

// Save ePrescription
router.post('/eprescriptions', async (req, res) => {
    const { doctorId, patientId, medicines, tests, notes, followUpDate } = req.body;

    if (!doctorId || !patientId) {
        return res.status(400).json({ message: "Doctor ID and Patient ID are required" });
    }

    try {
        // Create a new ePrescription document
        const newPrescription = new EPrescription({
            doctorId,
            patientId,
            medicines,
            tests,
            notes,
            followUpDate,
        });

        // Save ePrescription
        const savedPrescription = await newPrescription.save();

        // Find the patient's appointment (using patientId instead of userId if patientId is the correct field)
        const appointment = await Appointment.findOne({ doctorId, patientId });
        if (!appointment) {
            return res.status(404).json({ message: "Appointment not found for the provided doctor and patient" });
        }

        // Trigger reminder logic here (e.g., create a reminder service or notification)

        res.status(201).json({
            message: "ePrescription created successfully",
            savedPrescription,
        });
    } catch (error) {
        console.error("Error creating ePrescription:", error);
        res.status(500).json({ message: "Server error", error: error.message });
    }
});

// Fetch ePrescriptions for a patient
router.get('/fetch/eprescriptions', async (req, res) => {
  const { patientId } = req.query;

  if (!patientId) {
      return res.status(400).json({ error: "Patient ID is required" });
  }

  try {
      const prescriptions = await EPrescription.find({ patientId });

      if (!prescriptions || prescriptions.length === 0) {
          return res.status(404).json({ error: "No prescriptions found" });
      }

      const formattedPrescriptions = prescriptions.map((prescription) => ({
          doctorId: prescription.doctorId, // Include doctorId if needed
          medicines: prescription.medicines.map((med) => ({
              name: med.name,
              dosage: med.dosage,
              duration: med.duration,
              instructions: med.instructions,
          })),
          tests: prescription.tests.map((test) => ({
              name: test.name,
              reason: test.reason,
              instructions: test.instructions,
          })),
          notes: prescription.notes,
          followUpDate: prescription.followUpDate,
          createdAt: prescription.createdAt.toISOString(), // Convert to ISO string for frontend compatibility
      }));

      res.json(formattedPrescriptions);
  } catch (error) {
      console.error("Error fetching prescriptions:", error);
      res.status(500).json({ error: "Error fetching prescriptions" });
  }
});


module.exports = router;
