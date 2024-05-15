// Plugins
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout'; 
// Import the styles
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';
// Worker
import { Worker } from '@react-pdf-viewer/core'; // install this library

import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import './index.css';


function App() {
  // for onchange event
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfFileError, setPdfFileError] = useState('');

  // for submit event
  const [viewPdf, setViewPdf] = useState(null);

  // Create the instance of defaultLayoutPlugin
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  // onchange event
  const fileType = ['application/pdf'];
  const handlePdfFileChange = (e) => {
    let selectedFile = e.target.files[0];
    if(selectedFile) {
      if(selectedFile && fileType.includes(selectedFile.type)) {
        let reader = new FileReader();
        reader.onloadend = () => {
          setPdfFile(reader.result);
          setPdfFileError('');
        };
        reader.readAsDataURL(selectedFile);
      } else {
        setPdfFile(null);
        setPdfFileError('Please select a valid PDF file');
      }
    } else {
      console.log('Select your file');
    }
  };

  // Form submit
  const handlePdfFileSubmit = (e) => {
    e.preventDefault();
    if(pdfFile !== null) {
      setViewPdf(pdfFile);
    } else {
      setViewPdf(null);
    }
  };

  return (
    <div className="container">
      <form className='form-group' onSubmit={handlePdfFileSubmit}>
        <br />
        <input type='file' className='form-control' required onChange={handlePdfFileChange} />
        {pdfFileError && <div className='error-msg'>{pdfFileError}</div>}
        <br />
        <br />
        <button type='submit' className='btn btn-success btn-lg'>
          UPLOAD
        </button>
      </form>
      <br />
      <h4>View PDF</h4>
      <div className='pdf-container'>
        {/* Show pdf */}
        {viewPdf && (
          <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.4.120/build/pdf.worker.min.js">
            <Viewer fileUrl={viewPdf} plugins={[defaultLayoutPluginInstance]} />
          </Worker>
        )}
        {/* If we don't have pdf or viewPdf state is null */}
        {!viewPdf && <>No PDF file selected</>}
      </div>
    </div>
  );
}

export default App;

