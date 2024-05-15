import React, { useRef } from 'react';
import './Feedback.css';
// import next_icon from '../../assets/next-icon.png';
// import back_icon from '../../assets/back-icon.png';
import user_1 from '../../assets/user-1.png';
import user_2 from '../../assets/user-2.png';
import user_3 from '../../assets/user-3.png';
// import user_4 from '../../assets/user-4.png';

const Feedback = () => {
  const slider = useRef();
  let tx = 0;

  const slideForward = () => {
    if (tx > -50) {
      tx -= 23;
    }
    slider.current.style.transform = `translateX(${tx}%)`;
  };

  const slideBackward = () => {
    if (tx < 0) { // Adjusted condition
      tx += 23; // Adjusted sliding amount
    }
    slider.current.style.transform = `translateX(${tx}%)`;
  };

  return (
    <div className='feedback'>
      {/* <img src={next_icon} alt='' className='next-btn' onClick={slideForward} />
      <img src={back_icon} alt='' className='back-btn' onClick={slideBackward} /> */}
      <div className='slider' ref={slider}>
        <ul>
          <li>
            <div className='slide'>
              <div className='user-info'>
                <img src={user_1} alt='' />
                <div>
                  <h3>Hasini Tharupa</h3>
                  <span>Galle </span>
                  <p>
                    Thank you for uploading your website. We appreciate your effort in making your content accessible
                    online. Your dedication to enhancing the online experience is commendable.
                  </p>
                </div>
              </div>
            </div>
            <div className='slide'>
              <div className='user-info'>
                <img src={user_2} alt='' />
                <div>
                  <h3>Punethra Karunarathna</h3>
                  <span>Galle </span>
                  <p>
                    Your website provides valuable information and services to users. Keep up the great work, and feel
                    free to reach out if you need any assistance or support with your website.
                  </p>
                </div>
              </div>
            </div>
            {/* <div className='slide'>
              <div className='user-info'>
                <img src={user_4} alt='' />
                <div>
                  <h3>Tharuka</h3>
                  <span>Colombo, Pannipitiya </span>
                  <p>
                    Your website provides valuable information and services to users. Keep up the great work, and feel
                    free to reach out if you need any assistance or support with your website.
                  </p>
                </div>
              </div>
            </div> */}
            <div className='slide'>
              <div className='user-info'>
                <img src={user_3} alt='' />
                <div>
                  <h3>Nipuni</h3>
                  <span>Matara </span>
                  <p>
                   An online learning platform provides valuable educational resources and services to users, allowing them to enhance their knowledge and skills in various subjects. Similar to traditional classroom learning, these platforms offer courses, tutorials, and interactive learning materials delivered through digital means.
                  </p>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default Feedback;
