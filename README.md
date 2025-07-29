<<<<<<< HEAD
# Fake Job Offer Detector

An AI-powered web application to detect fraudulent job offers and protect job seekers from scams.

## Features

- **Upload & Analyze**: Just upload a job offer document (PDF, image, DOCX) and get instant analysis
- **No Manual Entry**: No need to fill out any form fields—just upload and get results
- **Text Extraction**: Extracts text from PDFs and images (OCR); DOCX placeholder
- **AI Analysis**: Analyzes the extracted text for fraud risk (mocked for now)
- **Modern UI**: Beautiful, responsive interface built with React
- **Risk Assessment**: Provides confidence scores and identifies potential risk factors
- **Smart Recommendations**: Offers actionable advice to verify job offers

## Technology Stack

- **Frontend**: React 18, HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Custom CSS with modern design patterns
- **Icons**: Lucide React for beautiful, consistent icons
- **OCR & PDF**: Tesseract.js for image OCR, pdfjs-dist for PDF text extraction

## Project Structure

```
fake-job-offer-detector/
├── public/
│   └── index.html
├── src/
│   ├── App.js          # Main application component
│   ├── App.css         # Component-specific styles
│   ├── index.js        # React entry point
│   └── index.css       # Global styles
├── package.json        # Dependencies and scripts
└── README.md           # Project documentation
```

## Getting Started

### Prerequisites

- Node.js (version 14 or higher)
- npm or yarn package manager

### Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   npm install
   ```
3. **Start the development server**:
   ```bash
   npm start
   ```
4. **Open your browser** and navigate to `http://localhost:3000`

## Usage

1. **Upload a file** (PDF, image, DOCX)
2. **Wait for extraction and analysis** (no manual input required)
3. **Review the results**:
   - Legitimacy prediction (Legitimate/Fraudulent/Suspicious)
   - Confidence score
   - Identified risk factors
   - Safety recommendations

## Current Implementation

The current version includes:
- ✅ Upload-only workflow (no manual form fields)
- ✅ File upload and text extraction (PDF, image; DOCX placeholder)
- ✅ Automatic analysis after extraction
- ✅ Responsive design for all devices
- ✅ Mock analysis functionality
- ✅ Beautiful result display with confidence indicators

## Next Steps

To complete the full application, you'll need to:
1. **Backend Integration**: Connect to a Python backend with the ML model
2. **API Development**: Create endpoints for job offer analysis
3. **Model Integration**: Integrate the fraud detection model from your dataset
4. **Database**: Add storage for analysis history and user accounts
5. **Authentication**: Add user registration and login features

## Backend Integration

The frontend is ready to connect to a backend API. The extracted text will be sent for analysis.

Expected API response format:
```javascript
{
  prediction: "legitimate" | "fraudulent" | "suspicious",
  confidence: number (0-100),
  riskFactors: string[],
  recommendations: string[]
}
```

## Contributing
=======
# 🛡️ AI-Powered Fake Job Offer Detector

A comprehensive web application that uses advanced machine learning to detect fake job offers. The application can analyze text, PDF files, DOCX documents, and images to identify potential job scams with detailed explanations and confidence scores.

## 🚀 Features

- **🤖 AI-Powered Analysis**: Uses a trained Logistic Regression model with TF-IDF vectorization
- **📁 Multi-Format Support**: Upload PDF, DOCX, TXT files, or images (JPG, PNG, BMP, TIFF)
- **📝 Text Input**: Paste job offer text directly for instant analysis
- **🔍 Fraud Indicators**: Detects 7 different types of fraud indicators
- **📊 Detailed Results**: Shows confidence scores, explanations, and risk factors
- **🎨 Modern UI**: Beautiful, responsive design with drag-and-drop file upload
- **📱 Mobile Friendly**: Works seamlessly on desktop and mobile devices
- **🔒 Security**: Server-side validation and secure file handling

## 🏗️ Architecture

### Backend (Flask + Scikit-learn)
- **Framework**: Flask with CORS support
- **ML Model**: Logistic Regression with TF-IDF Vectorizer
- **File Processing**: PDF, DOCX, and image OCR support
- **Data Processing**: Pandas for data handling
- **Model Storage**: Joblib for model persistence

### Frontend (React)
- **Framework**: React 18 with modern hooks
- **File Upload**: React Dropzone for drag-and-drop functionality
- **Styling**: Modern CSS with gradients and animations
- **HTTP Client**: Axios for API communication
- **Icons**: Lucide React for beautiful icons

## 📁 Project Structure

```
ai-job-offer-detector/
├── backend.py                    # Flask backend application
├── requirements.txt              # Python dependencies
├── combined_offer_fraud_dataset.csv  # Training dataset
├── models/                       # Trained model files (auto-generated)
│   ├── fake_offer_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_info.pkl
├── uploads/                      # Temporary file uploads
├── frontend/                     # React frontend
│   ├── package.json
│   ├── public/
│   │   ├── index.html
│   │   └── manifest.json
│   └── src/
│       ├── App.js               # Main React component
│       ├── App.css              # Styling
│       ├── index.js
│       └── index.css
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Node.js 14+
- npm or yarn

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask backend**:
   ```bash
   python backend.py
   ```
   
   The backend will:
   - Load the dataset from `combined_offer_fraud_dataset.csv`
   - Train the ML model automatically on first run
   - Save the trained model to `models/` directory
   - Start the Flask server on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```
   
   The frontend will start on `http://localhost:3000`

## 🎯 Usage

1. **Open the application** in your browser at `http://localhost:3000`

2. **Analyze job offers** using either method:
   - **File Upload**: Drag and drop or click to upload files (PDF, DOCX, TXT, images)
   - **Text Input**: Paste job offer text directly into the text area

3. **View Results**: The application will display:
   - **Prediction**: "Real" or "Fake" with color-coded badges
   - **Confidence Score**: Percentage indicating model confidence
   - **Fraud Indicators**: 7 different types of detected fraud patterns
   - **AI Explanation**: Detailed explanation of the prediction
   - **Text Preview**: Extracted text from uploaded files

## 🔧 API Endpoints

### POST `/predict`
Analyzes job offer text and returns prediction results.

**Request Body**:
```json
{
  "text": "Your job offer text here..."
}
```

**Response**:
```json
{
  "text": "Original text",
  "prediction": "Real",
  "confidence": 0.85,
  "confidence_percentage": "85.0%",
  "fraud_indicators": {
    "urgent_language": 0,
    "payment_requests": 0,
    "personal_info_requests": 0,
    "unrealistic_offers": 0,
    "poor_grammar": 0,
    "generic_company_info": 0,
    "suspicious_contact": 0
  },
  "explanation": [
    "✅ This job offer appears to be GENUINE based on our AI analysis.",
    "📋 Detailed offer letter with comprehensive information.",
    "⏰ No pressure tactics or urgent language detected.",
    "💼 No requests for payment or fees.",
    "💡 Recommendation: Proceed with normal application process, but always verify the company independently.",
    "🎯 High confidence level (85.0%) in this prediction."
  ],
  "text_length": 450,
  "model_accuracy": 0.97
}
```

### POST `/upload`
Upload and analyze a file.

**Request**: Multipart form data with file

**Response**: Same as `/predict` with additional file information

### GET `/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_accuracy": 0.97
}
```

### GET `/model-info`
Get model performance information.

**Response**:
```json
{
  "accuracy": 0.97,
  "precision_fake": 0.98,
  "recall_fake": 0.95,
  "f1_fake": 0.96,
  "precision_real": 0.96,
  "recall_real": 0.99,
  "f1_real": 0.97,
  "total_samples": 17889,
  "training_samples": 14311,
  "test_samples": 3578
}
```

## 🤖 Machine Learning Model

### Model Details
- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF Vectorizer with bigrams
- **Features**: 5000 most frequent terms
- **Training**: Automatic on first run using the provided dataset
- **Performance**: High accuracy in distinguishing between real and fraudulent offers

### Fraud Indicators Detected
1. **Urgent Language**: Pressure tactics and time-sensitive demands
2. **Payment Requests**: Requests for fees, deposits, or transfers
3. **Personal Info Requests**: Early requests for sensitive information
4. **Unrealistic Offers**: Promises of unrealistic salaries or benefits
5. **Poor Grammar**: Suspicious writing quality
6. **Generic Company Info**: Lack of specific company details
7. **Suspicious Contact**: Use of personal emails, WhatsApp, etc.

## 🎨 UI Features

- **Modern Design**: Gradient backgrounds and glassmorphism effects
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Interactive Elements**: Hover effects, loading animations, and smooth transitions
- **Color-coded Results**: Green for real offers, red for fake offers
- **Progress Indicators**: Visual confidence bars and loading spinners
- **Drag & Drop**: Intuitive file upload interface

## 🔒 Security Features

- **Input Validation**: Server-side validation of all inputs
- **File Type Validation**: Strict file type checking
- **File Size Limits**: 16MB maximum file size
- **Temporary Storage**: Files are deleted after processing
- **Error Handling**: Graceful error handling and user-friendly messages
- **CORS Support**: Cross-origin resource sharing enabled for development

## 🚀 Deployment

### Backend Deployment
The Flask app can be deployed to:
- Heroku
- AWS Elastic Beanstalk
- Google Cloud Platform
- Any Python-compatible hosting service

### Frontend Deployment
The React app can be deployed to:
- Netlify
- Vercel
- GitHub Pages
- AWS S3 + CloudFront

## 🤝 Contributing
>>>>>>> 48562def39d37cc7a9f698a2417c4f337ac070b9

1. Fork the repository
2. Create a feature branch
3. Make your changes
<<<<<<< HEAD
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the SAP Hackfest - AI-Powered fake job offer letter detector.

## Support

For questions or support, please refer to the project documentation or contact the development team. 

If your React app at [http://localhost:3000](http://localhost:3000) is not loading, here are the steps to diagnose and fix the issue:

---

### 1. **Check Terminal Output**
- Look at your terminal where you ran `npm start`.
- Are there any errors or warnings?  
  If yes, please copy and share them here.

---

### 2. **Check for Compilation Errors**
- If there are errors like "Failed to compile" or missing files, the app will not start.
- If you see a message like "Compiled successfully!", the app should be running.

---

### 3. **Check for Port Conflicts**
- If another app is using port 3000, React may prompt you to use another port or fail to start.
- Try stopping all other local servers and run `npm start` again.

---

### 4. **Try Restarting the Server**
- Stop the current process (press `Ctrl+C` in the terminal).
- Run:
  ```sh
  npm start
  ```
- Watch for any errors in the terminal.

---

### 5. **Check Browser/Network**
- Make sure you are opening the correct URL: [http://localhost:3000](http://localhost:3000)
- Try a different browser or incognito mode.

---

### 6. **Check for Missing Dependencies**
- If you see errors about missing modules, run:
  ```sh
  npm install
  ```

---

**Please let me know what you see in your terminal after running `npm start`. If there are any error messages, paste them here and I’ll help you fix them!** 
=======
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check the console for error messages
2. Ensure all dependencies are installed
3. Verify the dataset file is present
4. Check that both backend and frontend servers are running
5. Ensure you have sufficient disk space for model training

## 🔮 Future Enhancements

- **Advanced ML Models**: Integration with BERT or other transformer models
- **Batch Processing**: Analyze multiple offers simultaneously
- **User Accounts**: Save analysis history and preferences
- **API Rate Limiting**: Production-ready API with rate limiting
- **Model Retraining**: Automatic model retraining with new data
- **Export Features**: Export analysis results to PDF/CSV
- **Real-time Updates**: WebSocket support for real-time analysis
- **Multi-language Support**: Support for multiple languages

## 📊 Model Performance

The model is trained on a comprehensive dataset of genuine and fake job offers, achieving:
- **Overall Accuracy**: 97%
- **Precision (Fake Detection)**: 98%
- **Recall (Fake Detection)**: 95%
- **Precision (Real Detection)**: 96%
- **Recall (Real Detection)**: 99%

## 🎯 Use Cases

- **Job Seekers**: Verify job offers before responding
- **HR Professionals**: Screen incoming applications
- **Recruitment Agencies**: Validate job postings
- **Educational Institutions**: Train students about job scams
- **Law Enforcement**: Identify patterns in job fraud schemes

## Deployment Instructions

### 1. Push to GitHub
- Initialize a git repo if you haven't:
  ```sh
  git init
  git add .
  git commit -m "Initial commit"
  git branch -M main
  git remote add origin <your-repo-url>
  git push -u origin main
  ```

### 2. Deploy Backend (Flask) on Render
- Go to [https://render.com/](https://render.com/) and sign up/log in.
- Click "New Web Service" and connect your GitHub repo.
- Set build command: `pip install -r requirements.txt`
- Set start command: `python backend.py`
- Set environment: Python 3.9+.
- Add environment variables if needed.
- After deploy, note your backend URL (e.g., `https://your-backend.onrender.com`).

### 3. Deploy Frontend (React) on Netlify
- Go to [https://netlify.com/](https://netlify.com/) and sign up/log in.
- Click "Add new site" > "Import an existing project" from GitHub.
- Set build command: `npm run build`
- Set publish directory: `frontend/build`
- In Netlify site settings, add an environment variable:
  - `REACT_APP_API_URL` = `https://your-backend.onrender.com`
- Redeploy the site after setting the variable.

### 4. Update Frontend API URLs
- In `frontend/src/App.js` (or wherever API calls are made), use:
  ```js
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
  ```
- Use `API_URL` for all axios/fetch calls.

### 5. Test Production
- Visit your Netlify site and test uploads/text analysis.
- The frontend will call your Render backend.

---

## Notes
- The backend must be running and publicly accessible for the frontend to work.
- For persistent user data collection, ensure `user_submissions.csv` is not in `.gitignore` if you want to keep it on the server (but don't push to GitHub).
- For privacy, add a notice if collecting user data.

---

**Protect yourself from job scams with AI-powered detection! 🛡️** 
>>>>>>> 48562def39d37cc7a9f698a2417c4f337ac070b9
