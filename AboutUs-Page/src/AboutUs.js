import React from 'react';
import './AboutUs.css';

const Team = () => {
  const teamMembers = [
    {
      id: 1,
      name: 'Imashi Thakshila',
      imgSrc: 'Imashi.jpg',
      position:'Leader',
      bio:'Handled backend and train the model',
    },
    {
      id: 2,
      name: 'Tharushi Dilesha',
      imgSrc: 'Tharushi.jpg',
      bio:'Handled backend and did the backend',
    },
    {
      id: 3,
      name: 'Akash Perera',
      imgSrc: 'Akash.jpg',
      bio:'Implemented the profile',
    },
    {
      id: 4,
      name: 'Hashini Tharupa',
      imgSrc: 'Tharupa.jpg',
      bio:'Created main pages and implemented customize calender',
    },
    {
      id: 5,
      name: 'Nipuni Pathirana',
      imgSrc: 'Nipuni.jpg',
      bio:'Created main pages and implemented book reccomandation',
    },
  ];


  return (
    <div className="team">
      <h2>Meet Our Team</h2>
      <div className="team-members">
        {teamMembers.map((member) => (
          <div key={member.id} className="team-member">
            <img style={{ width: '150px', height: '180px' }} src={member.imgSrc} alt={member.name} />
            <h3>{member.name}</h3>
            <p>{member.position}</p>
            <p>{member.bio}</p>
          </div>
        ))}
      </div>
    </div>
  );
};


export default Team