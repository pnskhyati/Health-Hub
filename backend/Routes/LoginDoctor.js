const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const router = express.Router();
const Doctor = require("../models/DoctorsSchema");
const DoctorLogin = require("../models/Doctors");
const { JWT_SECRET } = require("../config");
const mongoose = require("mongoose");


// Signup Route
router.post("/signup", async (req, res) => {
    const { name, email, password, specialization } = req.body;

    try {
        // Check if the email already exists in doctorsLogins
        const existingUser = await DoctorLogin.findOne({ email });
        if (existingUser) {
            return res.status(409).json({ message: "User already exists. Please Login." });
        }

        // Generate a unique ID
        const doctorId = new mongoose.Types.ObjectId();

        // Hash the password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Save to doctorsLogins
        const newDoctorLogin = new DoctorLogin({
            _id: doctorId, // Set the same ID
            name,
            email,
            password: hashedPassword,
        });
        await newDoctorLogin.save();

        // Save to +RegisteredDoctors
        const newRegisteredDoctor = new Doctor({
            _id: doctorId, // Use the same ID
            "Doctor's Name": name,
            speciality: specialization,
        });
        await newRegisteredDoctor.save();

        res.status(201).json({ message: "Signup Successful" });
    } catch (error) {
        console.log("Error during signup:", error);
        res.status(500).json({ message: "Server Error", error: error.message });
    }
});


// Login Route
router.post('/login', async (req, res) => {
    const { email, password } = req.body;
    try {
        const user = await DoctorLogin.findOne({ email });
        if (!user) {
            return res.status(401).json({ message: "Invalid Credentials" });
        }

        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            return res.status(401).json({ message: "Invalid Credentials" });
        }

        const token = jwt.sign({ _id: user._id }, JWT_SECRET, { expiresIn: '1h' });
        res.status(200).json({
            message: 'Login successful',
            token,
            user: { name: user.name, email: user.email, _id: user._id }
        });
    } catch (error) {
        console.error('Error during login:', error);
        res.status(500).json({ message: 'Server Error', error: error.message });
    }
});

module.exports = router;
