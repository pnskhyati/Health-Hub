const mongoose = require('mongoose');

const ePrescriptionSchema = new mongoose.Schema({
    doctorId: { type: mongoose.Schema.Types.ObjectId, ref: 'Doctor', required: true },
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient', required: true },
    medicines: [
        {
            name: String,
            dosage: String,
            duration: String,
            instructions: String,
        },
    ],
    tests: [
        {
            name: String,
            reason: String,
        },
    ],
    notes: { type: String, required: false },
    createdAt: { type: Date, default: Date.now },
});

const EPrescription = mongoose.model('EPrescription', ePrescriptionSchema);

module.exports = EPrescription;
