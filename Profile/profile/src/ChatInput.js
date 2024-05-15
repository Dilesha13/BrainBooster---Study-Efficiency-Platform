import React, { useState } from 'react';
import Attach from './attach.png';
import Img from './img.png';

const ChatInput = ({ onSubmit }) => {
  const [text, setText] = useState('');
  const [img, setImg] = useState(null);

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleSend = () => {
    onSubmit(text, img);
    setText('');
    setImg(null);
  };

  const handleFileChange = (e) => {
    setImg(e.target.files[0]);
  };

  return (
    <div className="input">
      <input
        type="text"
        placeholder="Type something..."
        onChange={handleTextChange}
        value={text}
      />
      <div className="send">
        <img src={Attach} alt="Attach" />
        <input
          type="file"
          style={{ display: "none" }}
          id="file"
          onChange={handleFileChange}
        />
        <label htmlFor="file">
          <img src={Img} alt="Img" />
        </label>
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
};

export default ChatInput;
