import React, { useState } from 'react';
import './LoginSignup.css'; // Assuming LoginSignup.css is in the same directory
import user_icon from '../../assets/person.png'; // Import user icon
import email_icon from '../../assets/email.png'; // Import email icon
import password_icon from '../../assets/password.png'; // Import password icon

const LoginSignup = () => {

  const [action, setAction] = useState("Sign up");

  return (
    <div className='container'>
      <div className='header'>
        <div className='text'>{action}</div>
        <div className='underline'></div>
      </div>
      <div className='inputs'>
        {action==="Login"?<div></div>:<div className='inputs'>
          <img src={user_icon} alt="User Icon" />
          <input type="text"  placeholder='Name'/>
        </div>}
        {/* Input with user icon */}
        
        {/* Input with email icon */}
        <div className='inputs'>
          <img src={email_icon} alt="Email Icon" />
          <input type="email" placeholder='Email Id' />
        </div>
        {/* Input with password icon */}
        <div className='inputs'>
          <img src={password_icon} alt="Password Icon" />
          <input type="password" placeholder='Password' />
        </div>
      </div>
      {action === "Sign Up" ? <div></div> : <div className='forget-password'>Lost Password ? <span>Try Again</span> </div>}
      <div className='submit-container'>
        <div className={action === "Sign up" ? "submit gray" : "submit"} onClick={() => { setAction("Sign up") }}>Sign Up</div>
        <div className={action === "Login" ? "submit gray" : "submit"} onClick={() => { setAction("Login") }}>Login</div>
      </div>
    </div>
  );
};

export default LoginSignup;
