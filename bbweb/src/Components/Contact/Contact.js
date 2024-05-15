import React from 'react'
import './Contact.css' // Assuming you have a CSS file for styling
import msg_icon from '../../assets/msg-icon.png' // Assuming image paths are correct
import mail_icon from '../../assets/mail-icon.png'
import phone_icon from '../../assets/phone-icon.png'
import location_icon from '../../assets/location-icon.png'
import white_arrow from '../../assets/white-arrow.png'

const Contact = () => {

  const [result, setResult] = React.useState("");

  const onSubmit = async (event) => {
    event.preventDefault();
    setResult("Sending....");
    const formData = new FormData(event.target);

    formData.append("access_key", "c74d98f0-c793-4bd4-b44d-1e79d9dc1f09"); 

    const response = await fetch("https://api.web3forms.com/submit", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.success) {
      setResult("Form Submitted Successfully");
      event.target.reset();
    } else {
      console.log("Error", data);
      setResult(data.message);
    }
  };

  return (
    <div className='contact'>
      <div className='contact-col'>
        <h3>Send us a Message <img src={msg_icon} alt="Message Icon"/> </h3>
        <p>
        Please don't hesitate to reach out to us. We're here to 
        assist you with any questions or concerns you may have about 
        our PDF reader website. Contact us via email, phone, or our 
        online form, and our team will be happy to help. </p>
        <ul>
            <li> <img src={mail_icon} alt="Email Icon"/>Contact@Brainbooster.com</li>
            <li> <img src={phone_icon} alt="Phone Icon"/> +(94) 77 1510 649</li>
            <li> <img src={location_icon} alt="Location Icon"/> No.27/3 madapathala, <br/>galle. </li>
        </ul>
      </div>
      <div className='contact-col'>
        <form onSubmit={onSubmit}> {/* Corrected syntax for event handler */}
            <label>Your Name</label>
            <input type='text' name='name' placeholder='Enter your name' required/>

            <label>Phone Number</label>
            <input type='tel' name='phone' placeholder='Enter your mobile Num.' required/>

            <label>Write your message here</label>
            <textarea name='message'rows='6' placeholder='Enter your messege' required></textarea>
            <button type='submit' className='btn dark-btn'>Submit<img src={white_arrow} alt="Arrow Icon"/></button>
        </form>
        <span>{result}</span>
      </div>
    </div>
  )
}

export default Contact
