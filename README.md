# brandoff

This project consists of two parts: front-end and back-end

Note: This project uses git large file storage to upload trained models and other big files. [More Info](https://git-lfs.com/)

## Steps to run the back-end
1. Change the current directory to the `server` folder
2. (Only for the first time) Create a new python virtual environment [Steps](https://docs.python.org/3/library/venv.html)
3. Activate the virtual environment
4. (Only for the first time)Install all the dependencies using
  ```pip install -r requirements.txt```
5. Start the server using ```uvicorn app:app --reload```
6. It should display the host address & port number on the command line. The back-end is ready to receive requests now.

## Steps to run the front-end
1. Make sure you have the latest version of NodeJS installed
2. Change the current directory to `client` folder.
3. (Only for the first time) Install all the npm dependencies using ```npm install```
4. Next, launch the front-end by running ```npm start```
5. Open the address ```localhost:3000``` on your browswer.
