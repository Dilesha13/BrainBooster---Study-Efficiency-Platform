import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Home from './Components/Home';
import QuizInstructions from './Components/quiz/QuizInstructions';
import Play from './Components/quiz/Play';
import QuizSummary from './Components/quiz/QuizSummary';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/play/instructions" element={<QuizInstructions />} />
        <Route path="/play/quiz" element={<Play />} />
        <Route path="/play/quizSummary" element={<QuizSummary />} />
        <Route path="/" element={<Home />} />
      </Routes>
    </Router>
  );
}

export default App;
