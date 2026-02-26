import React, { useState } from 'react';
import { FaAngleDown } from "react-icons/fa";
import { Link } from 'react-router-dom';
import { CiHome } from 'react-icons/ci';
import './MedicineCategories.css';

export default function MedicineCategories() {
  // State to manage the visibility of the subCategories
  const [showSubCategories, setShowSubCategories] = useState(false);

  const toggleSubCategories = () => {
    setShowSubCategories(!showSubCategories); // Toggle the visibility
  }

  return (
    <>
      <nav className='navbarEpharma'>
        <div className="container1 border1">
          <div className="row">
            <div className="col-sm-9 navPart2">
              <ul className="list list-inline services">
                <li className="list-inline-item">
                  <Link to="/order-medicines" className="list-element">
                    <CiHome /> Home
                  </Link>
                </li>

                <li className="list-inline-item">
                  <button
                    className={`allCategoriesTab ${showSubCategories ? 'active' : ''}`}
                    onClick={toggleSubCategories}
                  >
                    <span className='text'>Healthcare</span>
                    <span><FaAngleDown /></span>
                  </button>
                </li>

                {showSubCategories && (
                  <div className="subCategories show">
                    <Link to="/all-medicines"><button>Fitness Supplements</button></Link>
                    <Link to="/all-medicines"><button>Personal Care</button></Link>
                    <Link to="/all-medicines"><button>Home Care</button></Link>
                    <Link to="/all-medicines"><button>Health Care</button></Link>
                    <Link to="/all-medicines"><button>Elderly Care</button></Link>
                    <Link to="/all-medicines"><button>Stomach Care</button></Link>
                    <Link to="/all-medicines"><button>Baby Care</button></Link>
                    <Link to="/all-medicines"><button>Health Condition</button></Link>
                  </div>
                )}

                <li className="list-inline-item"><Link to="/all-medicines" className="list-element">Medicine</Link></li>
                <li className="list-inline-item"><Link to="/" className="list-element">Lab Tests</Link></li>
                <li className="list-inline-item"><Link to="/" className="list-element">Health Blogs</Link></li>
                <li className="list-inline-item"><Link to="/" className="list-element">Contact Us</Link></li>
              </ul>
            </div>
          </div>
        </div>
      </nav>
    </>
  );
}
