import React from 'react'
import './Hero.css'
import dark_arrow from '../../assets/dark-arrow.png'

const Hero = () => {
  return (
    <div className='hero container'>
      <div className='hero-text'>
        <h1>We Ensure better education for a better world</h1>
        <p>Welcome to our PDF reading website and instructional platform! Our goal is to provide you with a seamless experience for 
          accessing and interacting with PDF documents while also offering clear and helpful instructions.</p>
        <button className='btn'> Try Free <img src={dark_arrow} alt=''/> </button>
      </div>
    </div>
  )
}

export default Hero;
