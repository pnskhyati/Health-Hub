import React from "react";
import "../Home.css";
import "../../App.css"
import ShoppingHome from "./mini-components/ShoppingHome";
import EPharmaHome from "./mini-components/EPharmaHome";


export default function OrderMedicines() {
  return (
    <>
     <div className="shoppingHome" >
     <ShoppingHome/>
     </div>
     <EPharmaHome/>
     
     
    </>
  )
}
