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

1. Fork the repository
2. Create a feature branch
3. Make your changes
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