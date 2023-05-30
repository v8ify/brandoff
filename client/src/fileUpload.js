import React, { useState } from 'react';
import axios from 'axios';
import './fileUpload.css'

function FileUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const fileSelectedHandler = event => {
    setSelectedFile(event.target.files[0]);
  };

  const blurFileUploadHandler = () => {
    const formData = new FormData();
    formData.append('image', selectedFile, selectedFile.name);

    axios.post('http://localhost:8000/blur', formData, {
      onUploadProgress: progressEvent => {
        setUploadProgress(Math.round((progressEvent.loaded / progressEvent.total) * 100));
      },
      responseType: "blob"
    })
    .then(response => {
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'blur_result.jpg');
      document.body.appendChild(link);
      link.click();
    })
    .catch(error => {
      console.log(error);
    });
  };

  const replaceFileUploadHandler = () => {
    const formData = new FormData();
    formData.append('image', selectedFile, selectedFile.name);

    axios.post('http://localhost:8000/replace', formData, {
      onUploadProgress: progressEvent => {
        setUploadProgress(Math.round((progressEvent.loaded / progressEvent.total) * 100));
      },
      responseType: "blob"
    })
    .then(response => {
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'replace_result.jpg');
      document.body.appendChild(link);
      link.click();
    })
    .catch(error => {
      console.log(error);
    });
  };

  return (
    <div className='file-upload-container'>
        <div className='input-container'>
            <input type="file" onChange={fileSelectedHandler} />
        </div>
        <button onClick={blurFileUploadHandler}>Blur</button>
        <button onClick={replaceFileUploadHandler}>Replace</button>
      {uploadProgress > 0 && <progress value={uploadProgress} max="100" />}
    </div>
  );
}

export default FileUpload;
