import React from "react";
import "./Home.css";
import "../App.css"
import Navbar from "./Navbar";

function Home({user}) {
  return (
    <div className="App">
      {/* Header Section */}
     <Navbar user={user}/>

      
      <section className="hero-section">
        <div className="hero-content">
          <h1>Your All-in-One Digital Hospital</h1>
          <p>Transforming healthcare access with AI and technology.</p>
          <div className="hero-buttons">
            <button>Upload X-ray for AI Analysis</button>
            <button onClick={() => window.location.href = '/book'}>Book a Consultation</button>
            <button>Explore Features</button>
          </div>
        </div>
        <div className="hero-image">
          <img src="home.jpg" alt="Healthcare" />
        </div>
      </section>

      {/* Key Features Highlights */}
      <section className="features-section">
        <h2>Key Features</h2>
        <div className="features">
          <div className="feature">
            <h3>AI Disease Detection</h3>
            <p>Analyze X-rays and medical images instantly.</p>
          </div>
          <div className="feature">
            <h3>Telemedicine Services</h3>
            <p>Video consultations and online prescriptions.</p>
          </div>
          <div className="feature">
            <h3>Personalized Medication Schedule</h3>
            <p>Stay on track with reminders and planning.</p>
          </div>
          <div className="feature">
            <h3>Medical Records Management</h3>
            <p>Securely save and access all your medical reports.</p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="how-it-works">
        <h2>How It Works</h2>
        <div className="steps">
          <div className="step">
            <h3>Step 1</h3>
            <p>Upload your medical data or X-rays.</p>
          </div>
          <div className="step">
            <h3>Step 2</h3>
            <p>Get an AI analysis or consult a doctor.</p>
          </div>
          <div className="step">
            <h3>Step 3</h3>
            <p>Receive personalized insights or prescriptions.</p>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="testimonials">
        <h2>What Our Users Say</h2>
        <div className="testimonial">
          <p>“HealthHub made healthcare so accessible for me. Highly recommended!”</p>
          <p>- A Satisfied User</p>
        </div>
      </section>

      {/* Footer Section */}
      <footer>
        <div className="footer-links">
          <a href="#privacy">Privacy Policy</a>
          <a href="#terms">Terms & Conditions</a>
          <a href="#help">Help Center</a>
        </div>
        <div className="contact-info">
          <p>Email: support@healthhub.com</p>
          <p>Phone: +91-1234567890</p>
        </div>
        <div className="social-links">
          <a href="#twitter">Twitter</a>
          <a href="#facebook">Facebook</a>
          <a href="#instagram">Instagram</a>
        </div>
      </footer>
    </div>
  );
}

export default Home;
