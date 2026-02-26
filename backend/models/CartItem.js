const mongoose = require('mongoose');

const CartItemSchema = new mongoose.Schema({
  userId: { type: String, required: true }, // Unique identifier for the user
  items: [
    {
      _id: { type: String, required: true }, // Item ID
      MedicineName: { type: String, required: true },
      image: { type: String, required: true },
      MRP: { type: Number, required: true },
      quantity: { type: Number, default: 1 },
    },
  ],
});

module.exports = mongoose.model('CartItem', CartItemSchema);