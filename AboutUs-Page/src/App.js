import { useState } from "react";
import Logo from"./assets/Logo.png";

function App() {
  const [value, setValue] = useState(null);
  return (
    <div className='w-full bg-[#0f172a] h-full min-h-[100vh 
    py-4 
    px-4 
    md:px-20'>
      <div className="w-full">
        <div className="flex flex-row justify-between items-center w-full h-10 px-5 2xl:px-40">
          <h3 className="cursor-pointer text-3xl font-bold text-cyan-600">Summary!</h3>
          <a href="#" /*onClick={(e) => e.preventDefault()}*/>
            <img src={Logo} className="w-16 h-20.4 rounded-lg cursor-pointer" alt="logo"/>
          </a>
        </div>
        <div className="flex flex-col items-center justify-center mt-4 p-4">
          <h1 className="text-3xl text-white text-center leading-10 font-semibold">Summarizer with<br/><span className="text-5xl font-bold text-cyan-500">OpenAI</span></h1>
          <p className="mt-5 text-lg text-gray-500 sm:text-xl text-center max-w-2xl">Simply upload your document and get a quick summary using OpenAI GPT Summarizer </p>
        </div>
        <div className="flex flex-col w-full items-center justify-center mt-5  ">
          <textarea placeholder="Paste doc content here ..."
          rows={7} className="block w-full md:w-[650px] rounded-md border border-slate-700 bg-slate-800 p-2 text-sm shadow-lg font-medium text-white
          focus:border-gray-500 focus:outline-none focus:ring-0" onChange={(e) => setValue(e.target.value)}> 
          </textarea>
          <button className="mt-5 bg-blue-500 px-5 py-2 text-white text-md font-semibold cursor-pointer rounded-md">Submit</button>
        </div>
      </div>    
    </div>
  );
}

export default App;
