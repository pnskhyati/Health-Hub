const mongoose = require("mongoose");

const appointmentSchema = new mongoose.Schema({
  patientName: String,
  patientEmail: String,
  appointmentDate: String,
  appointmentTime: String,
  doctorId: mongoose.Schema.Types.ObjectId,
  doctorName: String,
  userId: { type: String, required: true }, // Ensure userId is a non-null string
});

module.exports = mongoose.model("Appointment", appointmentSchema);
