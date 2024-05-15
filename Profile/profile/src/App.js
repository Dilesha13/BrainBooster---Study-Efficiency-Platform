import React, { useRef, useState, useEffect } from 'react';
import './App.css';
import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth';
import { getFirestore, collection, addDoc, serverTimestamp, query, orderBy } from 'firebase/firestore';
import { useAuthState } from 'react-firebase-hooks/auth';
import { useCollectionData } from 'react-firebase-hooks/firestore';
import { getStorage, ref, uploadBytes, getDownloadURL } from 'firebase/storage'; // Import storage functions

// Firebase configuration object
const firebaseConfig = {
  apiKey: "AIzaSyBX1uu7oUaNZHg0WD6C9mls7SoHQTPIs98",
  authDomain: "profile-eb12f.firebaseapp.com",
  projectId: "profile-eb12f",
  storageBucket: "profile-eb12f.appspot.com",
  messagingSenderId: "300953728761",
  appId: "1:300953728761:web:f7295957bc4daad0375428",
  measurementId: "G-QMWJTB5KWK"
};
// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const firestore = getFirestore(app);
const messagesRef = collection(firestore, 'messages');
const orderedQuery = query(messagesRef, orderBy('createdAt', 'asc'));
const storage = getStorage(app); // Initialize storage

function App() {
  const [user] = useAuthState(auth);

  return (
    <div className="App">
      <header>
        <h1>Chat Room</h1>
        <SignOut />
      </header>

      <section>
        {user ? <ChatRoom /> : <SignIn />}
      </section>
    </div>
  );
}

function SignIn() {
  const signInWithGoogle = () => {
    const provider = new GoogleAuthProvider();
    signInWithPopup(auth, provider)
      .then((result) => {
        // Handle successful sign-in
        console.log("User signed in:", result.user.displayName);
      })
      .catch((error) => {
        // Handle errors
        console.error("Error signing in with Google:", error);
      });
  };

  return (
    <div className="sign-in-container">
      <button className="sign-in-button" onClick={signInWithGoogle}>Sign in with Google</button>
      <p className="sign-in-text">Tap into the wisdom of the web to clear doubts and find answers!!</p>
    </div>
  );
}

function SignOut() {
  return auth.currentUser && (
    <button className="sign-out" onClick={() => signOut(auth)}>Log Out</button>
  );
}

function ChatRoom() {
  const dummy = useRef();
  const messagesRef = collection(firestore, 'messages');
  const orderedQuery = query(messagesRef, orderBy('createdAt', 'asc'));
  const [messages] = useCollectionData(orderedQuery, { idField: 'id' });
  const [formValue, setFormValue] = useState('');
  const currentUser = auth.currentUser;

  useEffect(() => {
    dummy.current.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    const { uid, photoURL } = currentUser;

    if (formValue.trim() !== '') {
      try {
        await addDoc(messagesRef, {
          text: formValue,
          createdAt: serverTimestamp(),
          uid,
          photoURL
        });

        setFormValue('');
      } catch (error) {
        console.error('Error sending message:', error);
      }
    }
  };

  return (
    <div className="chat-room-container">
      <main>
        {messages &&
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`message ${
                msg.uid === currentUser.uid ? 'sent' : 'received'
              }`}
            >
              <img
                src={msg.photoURL || 'https://api.adorable.io/avatars/23/abott@adorable.png'}
                alt="Profile"
                className="avatar"
              />
              <div className="text-bubble">{msg.text}</div>
            </div>
          ))}
        <span ref={dummy}></span>
      </main>


      <form onSubmit={sendMessage}>
        <div className="input">
          <input
            type="text"
            placeholder="Type something..."
            value={formValue}
            onChange={(e) => setFormValue(e.target.value)}
          />
          <button type="submit">Send</button>
        </div>
      </form>
    </div>
  );
}

export default App;
