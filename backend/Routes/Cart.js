const express = require('express')
const CartItem = require('../models/CartItem')
const router = express.Router();

// Get Cart Items from the user

router.get('/:userId', async (req, res)=>{
    try{
        const { userId } = req.params;
        const cart = await CartItem.findOne({ userId });
        res.status(200).json(cart ? cart.items : []);
    } catch(error)
    {
        res.status(500).json({ error: 'Failed to fetch cart items.' });
    
    }
})

// Add or update cart items

router.post('/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const { items } = req.body;
        let cart = await CartItem.findOne({ userId });
        if (cart) {
            // Update the cart
            cart.items = items;
          } else {
            // Create a new cart
            cart = new CartItem({ userId, items });
          }
          await cart.save();
          res.status(200).json({ message: 'Cart updated successfully.' });
          
    }
    catch(error)
    {
        res.status(500).json({ error: 'Failed to update the cart.' });
    }
})

// Remove an item from the cart
router.delete('/:userId/:itemId', async (req, res) => {
    try {
      const { userId, itemId } = req.params;
      const cart = await CartItem.findOne({ userId });
  
      if (cart) {
        cart.items = cart.items.filter((item) => item._id !== itemId);
        await cart.save();
      }
  
      res.status(200).json({ message: 'Item removed from cart.' });
    } catch (error) {
      res.status(500).json({ error: 'Failed to remove the item from the cart.' });
    }
  });
  
  module.exports = router;