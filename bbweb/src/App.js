import React, { useState } from 'react'
import Navbar from './Components/Navbar/Navbar'
import Hero from './Components/Hero/Hero'
import Program from './Components/Programs/Program'
import Title from './Components/Title/Title'
import About from './Components/About/About'
import Campus from './Components/Campus/Campus'
import Feedback from './Components/Feedback/Feedback'
import Contact from './Components/Contact/Contact'
import Footer from './Components/Footer/Footer'
import VideoPlayer from './Components/VideoPlayer/VideoPlayer'


const App = () => {
  const[playState, setPlayState] = useState(false);


  return (
    <div>
      <Navbar/>
      <Hero/>
      <div className='container'>
        <Title subTitle ='Our Program' title='Belongs lots of fields'/>
         <Program/>
         <About setPlayState={setPlayState}/>
         <Title subTitle ='Features' title=''/>
          <Campus/>
          <Title subTitle ='Feedback' title='What Students Says '/>
        <Feedback/>
        <Title subTitle ='Contact Us' title='Get in Touch '/>
        <Contact/>
        <Footer/>
      </div>
      <VideoPlayer playState={playState} setPlayState={setPlayState}/>
    </div>
  )
}

export default App

