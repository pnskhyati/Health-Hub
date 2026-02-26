import Home from './components/Home'
import Login from './components/Login'
import Signup from './components/Signup'
import { BrowserRouter, Route, Routes } from "react-router-dom"
import OrderMedicines from './components/E-Pharmacy/OrderMedicines'
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import { CartProvider } from './components/E-Pharmacy/Cart/CartContext';
import CartPage from './components/E-Pharmacy/Cart/CartPage';
import ProductDetails from './components/E-Pharmacy/AllProducts/ProductDetails';
import BookAppointments from './components/consultation/BookAppointments'
import RoomPage from './components/meeting/RoomPage'
import Checkout from './components/E-Pharmacy/CheckoutPage/Checkout'
import MedicineSchedule from './components/MedicineSchedule-Feature/MedicineSchedule'
import { useState } from 'react'
import MedicalRecords from './components/MedicalRecords/MedicalRecords'
import './App.css'
import DiseaseDetection from './components/Disease-Detection/DiseaseDetection'
import ChoicePage from './components/ChoicePage'
import DoctorSignup from './components/DoctorSignup'
import DoctorLogin from './components/DoctorLogin'
import DoctorDashboard from './components/DoctorsDashboard'
import NutritionChatbot from './components/Chatbot/NutritionChatbot'
import Pneumonia from './components/Disease-Detection/Pneumonia'
import AtriumSegmentation from './components/Disease-Detection/AtriumSegmenation'
import HeartDetection from './components/Disease-Detection/HeartDetection'


function App() {

  const [user, setUser] = useState(null);

  return (
   <CartProvider>
     <BrowserRouter>
        <Routes>
  

          <Route path="/" exact={true} element={<Home  user={user} />} />
          {/* <Route path={`/${userId}`} exact={true} element={<Home  user={user} />} /> */}
          <Route path="/order-medicines" exact={true} element={<OrderMedicines/>} />
          <Route path={`cart`} exact={true} element={<CartPage/>} />
          <Route path="/all-medicines" exact={true} element={<ProductDetails/>} />
          <Route path="/book" exact={true} element={<BookAppointments/>} />
          <Route path="/room/:roomId" element={<RoomPage />} />
          <Route path="/login-patient" exact={true} element={<Login setUser={setUser} />} />
          <Route path="/signup-patient" exact={true} element={<Signup  />} />
          <Route path="/checkout" element={<Checkout />} />
          <Route path="/medicine-schedule" element={<MedicineSchedule />} />
          <Route path="/medical-records" element={<MedicalRecords />} />
          <Route path="/disease-detection" element={<DiseaseDetection />} />
          <Route path="/choice-page" element={<ChoicePage />} />
          <Route path="/doctor-signup" element={<DoctorSignup />} />
          <Route path="/doctor-login" element={<DoctorLogin />} />
          <Route path="/doctors-dashboard" element={<DoctorDashboard />} />
          <Route path="/diet-plan" element={<NutritionChatbot />} />
          <Route path="/analyze/pneumonia" element={<Pneumonia />} />
          <Route path="/analyze/atrium-segmentation" element={<AtriumSegmentation />} />
          <Route path="/analyze/heart-detection" element={<HeartDetection />} />
        </Routes>
    </BrowserRouter>
   </CartProvider>
  )
}

export default App
