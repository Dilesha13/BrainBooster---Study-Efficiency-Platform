import React, {useState} from 'react';
import {Document, Page, pdfjs} from 'react-pdf';
import './PdfSummarizer.css';


pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.js`;
const PdfSummarizer = () => {
    const [file, setFile] = useState(null);
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(1);
    const [text, setText] = useState('');
    const [summary, setSummary] = useState('');
    const [title, setTitle] = useState('');
    const [author, setAuthor] = useState('');

    const onFileChange = async (event) => {
        const selectedFile = event.target.files[0];
        setFile(selectedFile);

        const reader = new FileReader();
        reader.onload = async (e) => {
            const buffer = e.target.result;
            const typedArray = new Uint8Array(buffer);
            const pdf = await pdfjs.getDocument(typedArray).promise;
            let fullText = '';

            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const pageText = await page.getTextContent();
                const pageString = pageText.items.map((item) => item.str).join('\n');
                fullText += pageString + '\n';
            }

            setText(fullText);

            const pdfInfo = await pdf.getMetadata();
            setTitle(pdfInfo.info.Title || 'Unknown Title');
            setAuthor(pdfInfo.info.Author || 'Unknown Author');
            setNumPages(pdf.numPages);
        };

        reader.readAsArrayBuffer(selectedFile);
    };
    
    const summarizeText = async () => {
        const sentences = text.split(/[.!?]/);
        const summaryText = sentences.slice(0, 3).join('. ') + '.';
        setSummary(summaryText);
    };

    return (
        <div className="container">
            <div className='w-full bg-[#0f172a] h-full min-h-[100vh] 
    py-4 
    px-4 
    md:px-20'>
      <div className="page">
        <div className="page">
          <h3>Summary!</h3>
          {/* <a href="#">
            <img src="./assets/Logo.png" alt="logo"/>
          </a> */}
        </div>
        <div className="page">
          <h1 className="text-3xl text-white text-center leading-10 font-semibold">Summarizer with<br/><span className="page1">BrainBooster</span></h1>
          <p className="page">Simply upload your document and get a quick summary using BrainBooster Summarizer.<br/></p>
        </div>
       </div>    
     </div>
            {/* <h1>PDF Summarizer</h1> */}
            <input type="file" onChange={onFileChange}/>
            {file && (
                <div className="pdf-info">
                    <h3>PDF Information</h3>
                    <p><strong>Title:</strong> {title}</p>
                    <p><strong>Author:</strong> {author}</p>
                    <p><strong>Number of Pages:</strong> {numPages}</p>
                    <Document file={file} onLoadSuccess={({numPages}) => setNumPages(numPages)}>
                        <Page pageNumber={pageNumber}/>
                    </Document>
                </div>
            )}
            <button className="button" onClick={summarizeText} disabled={!file}>Summarize</button>
            <div className="summary">
                <h2>Summary</h2>
                <p>{summary}</p>
            </div>
        </div>
    );
};

export default PdfSummarizer;