const mongoose = require('mongoose');

// Define the schema to match your existing collection structure
const MedicineSchema = new mongoose.Schema({
    Unnamed_0: Number,
    medicineName: String,  // Using a more standard field name
    composition: String,
    manufacturer: String,
    MRP: String,
    bestPrice: String,
    description: String,
    keyBenefits: String,
    ratings: String,
    reviews: String,
    howToUse: String,
    precautions: String,
    ingredients: String,
    quantity: String
}, {
    collection: 'Medicines'  // Explicitly specify the collection name
});

const Medicine = mongoose.model('Medicine', MedicineSchema);
module.exports = { Medicine };
