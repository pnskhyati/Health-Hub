import React, { useState } from 'react'
import "./ShoppingHome.css";
import { FaAngleDown } from "react-icons/fa";
import Dialog from '@mui/material/Dialog';
import Slide from '@mui/material/Slide';
import { FaSearch } from "react-icons/fa";
import { MdClose } from 'react-icons/md'

const Transition = React.forwardRef(function Transition(props,ref){
  return <Slide direction='up' ref={ref } {...props}/>
})
export default function CityDropdown() {
  
  const city = ["Pune", "Mumbai", "Kolhapur", "Bhubaneshwar"];
  const [cityList, setCityList] = useState(city);
  const [isOpenModel, setIsOpenModel] = useState(false);
  const [cityName, setCityName] = useState("Your Location")

  const filterList = (e) =>{
    const keyword = e.target.value;

    console.log(keyword);
    
    if(keyword !=="")
    {
      const list = city.filter((item)=>{
        return item.toLowerCase().includes(keyword);
      })
      setCityList(list);
    }
    else {
      setCityList(city);
    }
    
  }

  return (
    <>
       <button className="cityDropdown" onClick={()=> {setIsOpenModel(true)}}>
            <div className="info">
              <span>{cityName}</span>
            </div>
           <span className="ml-auto"> <FaAngleDown /></span>
          </button>

        <Dialog open={isOpenModel} className='locationModel' TransitionComponent={Transition}>
          <h3>Choose Your Delivery Location</h3>
          <p>Enter your address and we will specify the offer in your area.</p>
            <div className="headerSearch">
              <input type="text" placeholder="Search Your City" onChange={filterList}/>
              <button><FaSearch/></button>
            </div>
              <button className='close_' onClick={()=>{ setIsOpenModel(false)}}><MdClose/></button>

                  <ul className='cityList'>
                    {cityList.map((cityName)=>( <li key={cityName}><button onClick={
                      
                      ()=> {
                        setCityName(cityName);
                        setIsOpenModel(false)
                      }
                      }>{cityName}</button></li>))}
                    

                  </ul>
        </Dialog>
    </>
  )
}
