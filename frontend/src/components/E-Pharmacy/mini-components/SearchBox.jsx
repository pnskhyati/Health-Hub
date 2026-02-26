import React from 'react'
import { FaSearch } from "react-icons/fa";

export default function SearchBox() {
  return (
    <>
      <div className="headerSearch">
              <input type="text" placeholder="Search" />
              <button><FaSearch/></button>
        </div>
    </>
  )
}
