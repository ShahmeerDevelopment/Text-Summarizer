# PDF Document Summarization

This project provides a web application for summarizing the contents of PDF documents. It uses a React frontend for the user interface and a Flask backend for handling file uploads and text summarization. The summarization is powered by a pre-trained BART (Bidirectional and Auto-Regressive Transformers) model.

## Features

- **Multiple File Upload**: Users can upload multiple PDF documents for summarization.
- **Text Extraction**: Extracts text from PDF documents using PyMuPDF.
- **Text Summarization**: Summarizes the extracted text using a pre-trained BART model.
- **User-Friendly Interface**: Provides a clean and intuitive UI for uploading files and viewing the summary.

## Technologies Used

- **Frontend**: React, Material-UI
- **Backend**: Flask, Flask-CORS
- **Machine Learning**: Hugging Face's Transformers (BART model)
- **PDF Processing**: PyMuPDF

## Setup Instructions

### Prerequisites

- Node.js and npm
- Python 3.x
- pip (Python package installer)

### Frontend Setup

1. Navigate to the `frontend` directory:

   ```bash
   cd frontend
## Install the dependencies:

npm install

- Start the React application:

npm start
The application will run on http://localhost:3000.

## Backend Setup
- Navigate to the backend directory:

cd backend

- Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

- Install the dependencies:
bash
pip install -r requirements.txt

- Ensure you have the pre-trained BART model. Update the model_dir variable in the app.py to point to your model directory.

- Start the Flask application:

flask run --port 5004
The Flask server will run on http://localhost:5004.

## Usage

- Open the React application in your browser.
- Use the file input to select one or more PDF documents.
- Click the "Summarize" button to upload the files and get the summarized text.
- The summary will be displayed on the screen.
- 
## Project Structure

- frontend/: Contains the React application.
- backend/: Contains the Flask application.
- requirements.txt: Lists the Python dependencies.

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Acknowledgments

- Hugging Face for the Transformers library and the BART model.
- Material-UI for the React UI components.
- PyMuPDF for PDF text extraction.
