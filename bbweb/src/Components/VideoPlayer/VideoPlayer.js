// VideoPlayer.js
import React, { useRef } from 'react';
import './VideoPlayer.css';
import Video_degree from '../../assets/pexels-(1080p).mp4';

const VideoPlayer = ({ playState, setPlayState }) => {

    const player = useRef(null);

    const closePlayer = (e) => {
        if (e.target === player.current) {
            setPlayState(false);
        }
    }

    return (
        <div className={`video-player ${playState ? '' : 'hide'}`} ref={player} onClick={closePlayer}>
            {playState && (
                <video src={Video_degree} autoPlay muted controls></video>
            )}
        </div>
    );
}

export default VideoPlayer;
