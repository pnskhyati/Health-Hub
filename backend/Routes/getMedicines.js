const express = require('express');
const { Medicine } = require('../models/MedicineSchema');
const router = express.Router();

router.get('/medicines', async (req, res) => {
    try {
        console.log("Fetching medicines...");
        const medicines = await Medicine.find().limit(100);  // Correctly fetch medicines from the existing 'Medicines' collection
        if (!medicines || medicines.length === 0) {
            return res.status(404).json({ message: 'No medicines found' });
        }
        res.json(medicines);  // Send medicines as JSON response
    } catch (err) {
        console.error("Error fetching medicines:", err);
        res.status(500).json({ message: err.message });
    }
});

module.exports = router;
