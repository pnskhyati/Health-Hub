const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const AllMedicinesRouter = require('./Routes/getMedicines')
const db = require('./db')
const path = require("path");
const patientLoginRoutes = require('./Routes/LoginPatient')
const cartRoutes = require('./Routes/Cart');
const fetchSpeciality = require('./Routes/DoctorsAppointment');
const  doctorDashboard  = require('./Routes/DoctorsDashboard');
const doctorLoginRoutes = require('./Routes/LoginDoctor')
const ePrescription = require('./Routes/ePrescription')
const nutritionRoutes = require('./Routes/nutritionRoutes');



dotenv.config();

const server = express();



//Middleware
server.use(express.json());
server.use(cors({
  origin: '*', // For development only (change to your frontend URL in production)
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
}));



  //Routes
  server.use('/', AllMedicinesRouter );
  server.use('/api/patients', patientLoginRoutes);
  server.use('/api/cart', cartRoutes);
  server.use('/api', fetchSpeciality);
  server.use('/api/doctors', doctorLoginRoutes);
  server.use('/api/dashboard', doctorDashboard);
  server.use('/api', ePrescription);
  server.use('/api', nutritionRoutes);
  
  
server.listen(5000, ()=>{
    console.log("Server listening on Port: 5000");
    
})