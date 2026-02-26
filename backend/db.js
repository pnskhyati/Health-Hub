const mongoose = require('mongoose')
require("dotenv").config();


const dbURI = process.env.MONGODB_URI
if (!dbURI) {
    console.error("Error: MONGODB_URI is not defined in the environment variables.");
    process.exit(1); // Exit the process if URI is not provided
  }

  const db = mongoose.connection;

mongoose.connect(dbURI)
.then(() => {
  console.log("MongoDB Connected");

  // After successful connection, call the scheduleNotifications function
 // Start the scheduling of notifications
})
.catch((error) => {
  console.error("MongoDB Connection error:", error);
  process.exit(1); // Exit the process if MongoDB connection fails
});

module.exports = db;