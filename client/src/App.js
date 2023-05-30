import FileUpload from './fileUpload';
import './App.css';

function App() {
  return (
    <div className="App">
      <div className='App-logo'>BrandOff 1.0</div>
      <header className="App-header">
        <FileUpload />
        <p>
          Upload an image with a logo.
        </p>
      </header>
    </div>
  );
}

export default App;
