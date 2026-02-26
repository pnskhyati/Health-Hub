const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const router = express.Router();
const Patient = require('../models/Patients');
const { JWT_SECRET } = require('../config');

// POST api/patients/signup
router.post('/signup', async (req, res) => {
  const { fullName, email, password } = req.body;
  try {
    // Check if the user already exists
    const existingUser = await Patient.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ message: "User already exists. Please Login." });
    }

    // Hash the password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create a new user
    const newPatient = new Patient({ fullName, email, password: hashedPassword });
    await newPatient.save();

    res.status(201).json({ message: 'Signup Successful' });

  } catch (error) {
    res.status(500).json({ message: 'Server Error' });
  }
});

// POST api/patients/login
router.post('/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await Patient.findOne({ email });
    if (!user) {
      return res.status(401).json({ message: "Invalid Credentials" });
    }

    // Compare the provided password with the stored hashed password
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(401).json({ message: "Invalid Credentials" });
    }

    // Generate a JWT token
    const token = jwt.sign({ _id: user._id }, JWT_SECRET, { expiresIn: '1h' });

    // Send back the token and user data (excluding password)
    res.status(200).json({ 
      message: 'Login successful', 
      token, 
      user: { fullName: user.fullName, email: user.email, _id: user._id }
    });

  } catch (error) {
    res.status(500).json({ message: 'Server error', error });
  }
});

module.exports = router;
