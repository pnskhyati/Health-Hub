import React from 'react';
import { useCart } from './CartContext'; // Import the custom hook
import './CartPage.css';
import { FiPlus, FiMinus } from 'react-icons/fi'; // Import icons for increment and decrement
import { useNavigate } from 'react-router-dom';

const CartPage = () => {
  const { cartItems, removeFromCart, updateQuantity } = useCart();

  const handleIncrement = (id) => {
    const item = cartItems.find((item) => item._id === id);
    updateQuantity(id, item.quantity + 1);
  };

  const handleDecrement = (id) => {
    const item = cartItems.find((item) => item._id === id);
    if (item.quantity > 1) {
      updateQuantity(id, item.quantity - 1);
    }
  };

  const handleRemoveItem = (id) => {
    removeFromCart(id);
  };

  const navigate = useNavigate();
  return (
    <div className="cart-page">
      <h2>Your Cart</h2>
      <div className="cart-items">
        {cartItems.length === 0 ? (
          <p>Your cart is empty</p>
        ) : (
          cartItems.map((item) => (
            <div key={item._id} className="cart-item">
              <img src={item.image} alt={item.MedicineName} className="cart-item-image" />
              <div className="cart-item-info">
                <p className="cart-item-name">{item.MedicineName}</p>
                <p className="cart-item-price">₹ {item.MRP}</p>
                <div className="cart-item-quantity">
                  <label>Quantity:</label>
                  <div className="quantity-controls">
                    <button onClick={() => handleDecrement(item._id)} className="quantity-button">
                      <FiMinus />
                    </button>
                    <span className="quantity-display">{item.quantity}</span>
                    <button onClick={() => handleIncrement(item._id)} className="quantity-button">
                      <FiPlus />
                    </button>
                  </div>
                </div>
                <button onClick={() => handleRemoveItem(item._id)} className="remove-item-button">
                  Remove
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="cart-summary">
        <h3>Summary</h3>
        <p>Total Items: {cartItems.length}</p>
        <p>
          Total Price: ₹
          {cartItems
            .reduce((total, item) => total + parseFloat(item.MRP) * item.quantity, 0)
            .toFixed(2)}
        </p>
        <button className="checkout-button" onClick={() => navigate('/checkout')}>Proceed to Checkout</button>
      </div>
    </div>
  );
};

export default CartPage;
