import React from 'react';
import { Shield } from 'lucide-react';
import '../App.css';


const Header = () => (
  <header className="header">
    <div className="header-content">
      <div className="logo">
        <Shield className="logo-icon" />
        <h1>AI-Powered Fake Job/Internship Offer Detector</h1>
        <img src="https://cdn-icons-png.flaticon.com/512/3062/3062634.png" alt="AI Shield" style={{ width: '48px', height: '48px', animation: 'floatY 2.5s infinite alternate' }} />
      </div>
      <p className="subtitle">Detect fake job offers using advanced AI technology</p>
    </div>
  </header>
);

export default Header;
