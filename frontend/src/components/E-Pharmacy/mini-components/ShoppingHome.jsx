import React from 'react';
import { Link } from "react-router-dom";
import "./ShoppingHome.css";
import CityDropdown from './CityDropdown';
import SearchBox from './SearchBox';
import { FiUser } from "react-icons/fi";
import { IoBagOutline } from "react-icons/io5";
import MedicineCategories from './MedicineCategories';
import { useCart } from '../Cart/CartContext'; 

export default function ShoppingHome() {

  const { cartItems } = useCart();
  console.log('Cart Items:', cartItems);

  const userId = localStorage.getItem('userId');
  

  return (
    <>
      <header className="navbar">
        <div className="logoWrapper">
          <Link to="/">
            <div className="logo">HealthHub</div>
          </Link>
        </div>

        <div className="part2">
          <CityDropdown/>
            <SearchBox/>
            

            <div className="part3 d-flex align-items-center ml-auto tab">
              
              <button className="circle mr-3 profile"><FiUser/></button>
              <div className="ml-auto cartTab">
              <span className="price">  ₹{" "}{cartItems
                        .reduce((total, item) => {
                          const price = item.MRP ? parseFloat(item.MRP) : 0;
                          return total + price * (item.quantity || 1);
                        }, 0)
                        .toFixed(2)}
                 </span>
                <div className="position-relative ml-2">
                 <Link to={`/cart`}>
                 <button className="circle ml-2 bag bagBackground "><IoBagOutline/></button>
                 </Link>
                <span className="count d-flex align-items-center justify-content-center">{cartItems.length}</span>
                </div>

              </div>
            </div>

        </div>
      </header>

    <MedicineCategories/>

    </>
  );
}
