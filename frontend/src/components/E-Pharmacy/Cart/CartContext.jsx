import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

const userId = localStorage.getItem('userId');
// Create Cart Context
const CartContext = createContext();

// Create a custom hook to use CartContext
export const useCart = () => {
  return useContext(CartContext);
};

// CartProvider to wrap around the app
export const CartProvider = ({ children }) => {
  const [cartItems, setCartItems] = useState([]);

   useEffect(() => {
    // Fetch cart items on page load
    const fetchCart = async () => {
      const { data } = await axios.get(`http://localhost:5000/api/cart/${userId}`);
      setCartItems(data);
    };

    fetchCart();
  }, []);

  const addToCart = async (product) => {
    const updatedCart = [...cartItems];
    const itemIndex = updatedCart.findIndex((item) => item._id === product._id);

    if (itemIndex !== -1) {
      updatedCart[itemIndex].quantity += 1;
    } else {
      updatedCart.push({ ...product, quantity: 1 });
    }

    setCartItems(updatedCart);

    // Save to backend
    await axios.post(`http://localhost:5000/api/cart/${userId}`, { items: updatedCart });
  };


  
  // Remove item from the cart
  const removeFromCart = async (productId) => {
    const updatedCart = cartItems.filter((item) => item._id !== productId);
    setCartItems(updatedCart);

    // Update backend
    await axios.post(`http://localhost:5000/api/cart/${userId}`, { items: updatedCart });
  };

  // Update quantity
  const updateQuantity = async (productId, newQuantity) => {
    const updatedCart = cartItems.map((item) =>
      item._id === productId ? { ...item, quantity: newQuantity } : item
    );

    setCartItems(updatedCart);

    // Update backend
    await axios.post(`http://localhost:5000/api/cart/${userId}`, { items: updatedCart });
  };

  return (
    <CartContext.Provider value={{ cartItems, addToCart, removeFromCart, updateQuantity }}>
      {children}
    </CartContext.Provider>
  );
};
