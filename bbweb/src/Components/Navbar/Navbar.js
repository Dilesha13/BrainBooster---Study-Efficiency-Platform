import React, { useEffect, useState } from 'react'
import './Navbar.css'
import logo from '../../assets/logo.png'
import menu_icon from '../../assets/menu-icon.png'

import { Link } from 'react-scroll';

const Navbar = () => {

  const [sticky, setStikey] = useState(false);

  useEffect(()=>{
    window.addEventListener('scroll', ()=>{
      window.scrollY > 700 ? setStikey(true) : setStikey(false)
    })
  },[]);

  const[mobileMenu, setMobileMenu] =useState(false);
  const toggleMenu = ()=>{
    mobileMenu ? setMobileMenu(false) : setMobileMenu(true);
  }

  return (
    
    <nav className={`container ${sticky? 'dark-nav' : ''}`}>
        <img src={logo} alt='' className='logo'/>
        <ul className={mobileMenu? '':'hide-mobile-menu'}>
            <li><Link to='hero' smooth={true} offset={0} duration={500}>Home</Link></li>
            <li><Link to='program' smooth={true} offset={-210} duration={500}>Program</Link></li>
            <li><Link to='about' smooth={true} offset={-80} duration={500}>About Us</Link></li>
            <li><Link to='contact' smooth={true} offset={0} duration={500}>Contact Us</Link></li>
            <li><button className='btn'>LogIn</button></li>
        </ul>
        <img src={menu_icon} alt=''className='menu-icon' onClick={toggleMenu}/> 
    </nav>
  )
}

export default Navbar;