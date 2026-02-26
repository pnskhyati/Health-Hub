const mongoose = require("mongoose");

const doctorSchema = new mongoose.Schema({
  "Doctor's Name": String,
  speciality: String,
}, {
    collection: '+RegisteredDoctors'  // Explicitly specify the collection name
});

module.exports = mongoose.model("Doctor", doctorSchema);
