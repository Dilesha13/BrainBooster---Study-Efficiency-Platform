// About.js
import React from 'react';
import './About.css';
import about_img from '../../assets/about.jpeg';
import play_icon from '../../assets/play-icon.png';

const About = ({ setPlayState }) => {
  return (
    <div className='about'>
      <div className='about-left'>
        <img src={about_img} alt="" className='about-img' />
        <img src={play_icon} alt="" className='play-icon' onClick={() => setPlayState(true)} /> {/* Call setPlayState when the play button is clicked */}
      </div>
      <div className='about-right'>
        <h2>About PDF Uploader</h2>
        <h3>Tomorrow's Leaders Today</h3><br />
        <p>PDF (Portable Document Format) upload functionality is a feature commonly found in web applications, allowing users to upload PDF files from their local devices to the server. This feature finds widespread use in various applications such as document management systems, online forms, educational platforms, and more.</p>
          <button className='btnn'> Upload PDF </button>

      </div>
    </div>
  );
}

export default About;
