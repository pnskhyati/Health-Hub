const mongoose = require('mongoose');

const PatientSchema = new mongoose.Schema({
    fullName:String,
    email:String,
    password:String,
});

const Patient = mongoose.model('Patient', PatientSchema);

module.exports = Patient;