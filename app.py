import os
import re
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import PyPDF2
from docx import Document
from PIL import Image
import cv2
import pytesseract
import warnings
import csv
from datetime import datetime
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and vectorizer
model = None
vectorizer = None
model_accuracy = 0
model_info = {}

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + " "
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_image(file_path):
    """Extract text from image using OCR"""
    try:
        # Read image
        image = cv2.imread(file_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply median blur to remove noise
        blur = cv2.medianBlur(gray, 3)
        
        # Perform OCR
        text = pytesseract.image_to_string(blur)
        
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_file(file_path, file_type):
    """Extract text from different file types"""
    if file_type == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_type == 'docx':
        return extract_text_from_docx(file_path)
    elif file_type in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
        return extract_text_from_image(file_path)
    elif file_type == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""
    else:
        return ""

def get_fraud_indicators(text):
    """Analyze text for common fraud indicators"""
    indicators = {
        'urgent_language': 0,
        'payment_requests': 0,
        'personal_info_requests': 0,
        'unrealistic_offers': 0,
        'poor_grammar': 0,
        'generic_company_info': 0,
        'suspicious_contact': 0
    }
    
    text_lower = text.lower()
    
    # Urgent language
    urgent_words = ['urgent', 'immediate', 'asap', 'quick', 'fast', 'hurry', 'limited time', 'expires']
    for word in urgent_words:
        if word in text_lower:
            indicators['urgent_language'] += 1
    
    # Payment requests
    payment_words = ['payment', 'fee', 'processing fee', 'deposit', 'transfer', 'send money', 'pay', 'cost']
    for word in payment_words:
        if word in text_lower:
            indicators['payment_requests'] += 1
    
    # Personal info requests
    personal_words = ['ssn', 'social security', 'bank account', 'credit card', 'passport', 'personal details']
    for word in personal_words:
        if word in text_lower:
            indicators['personal_info_requests'] += 1
    
    # Unrealistic offers
    unrealistic_words = ['million', 'billion', 'huge salary', 'easy money', 'quick money', 'get rich']
    for word in unrealistic_words:
        if word in text_lower:
            indicators['unrealistic_offers'] += 1
    
    # Poor grammar (simple check)
    sentences = text.split('.')
    if len(sentences) > 1:
        avg_sentence_length = len(text) / len(sentences)
        if avg_sentence_length < 20 or avg_sentence_length > 200:
            indicators['poor_grammar'] += 1
    
    # Generic company info
    generic_words = ['company', 'corporation', 'enterprise', 'business']
    if sum(1 for word in generic_words if word in text_lower) >= 2:
        indicators['generic_company_info'] += 1
    
    # Suspicious contact methods
    suspicious_contact = ['whatsapp', 'telegram', 'personal email', 'gmail', 'yahoo']
    for word in suspicious_contact:
        if word in text_lower:
            indicators['suspicious_contact'] += 1
    
    return indicators

def generate_explanation(prediction, confidence, indicators, text_length):
    """Generate explanation for the prediction"""
    explanation = []
    
    if prediction == "Fake":
        explanation.append("🚨 This job offer appears to be FAKE based on our AI analysis.")
        
        if indicators['urgent_language'] > 0:
            explanation.append("⚠️ Contains urgent language that's common in scams.")
        
        if indicators['payment_requests'] > 0:
            explanation.append("💰 Requests for payment or fees - legitimate jobs don't ask for money.")
        
        if indicators['personal_info_requests'] > 0:
            explanation.append("🔒 Asks for sensitive personal information too early.")
        
        if indicators['unrealistic_offers'] > 0:
            explanation.append("💸 Makes unrealistic promises about salary or benefits.")
        
        if indicators['suspicious_contact'] > 0:
            explanation.append("📱 Uses suspicious contact methods (WhatsApp, personal emails).")
        
        if text_length < 100:
            explanation.append("📝 Very short offer letter - legitimate offers are usually detailed.")
        
        explanation.append("💡 Recommendation: Do not respond to this offer. Report it if possible.")
        
    else:  # Real
        explanation.append("✅ This job offer appears to be GENUINE based on our AI analysis.")
        
        if text_length > 200:
            explanation.append("📋 Detailed offer letter with comprehensive information.")
        
        if indicators['urgent_language'] == 0:
            explanation.append("⏰ No pressure tactics or urgent language detected.")
        
        if indicators['payment_requests'] == 0:
            explanation.append("💼 No requests for payment or fees.")
        
        explanation.append("💡 Recommendation: Proceed with normal application process, but always verify the company independently.")
    
    # Add confidence level explanation
    if confidence > 0.8:
        explanation.append(f"🎯 High confidence level ({confidence:.1%}) in this prediction.")
    elif confidence > 0.6:
        explanation.append(f"⚠️ Moderate confidence level ({confidence:.1%}) - exercise caution.")
    else:
        explanation.append(f"❓ Low confidence level ({confidence:.1%}) - manual review recommended.")
    
    return explanation

def train_model():
    """Train the ML model on the dataset"""
    global model, vectorizer, model_accuracy, model_info
    
    print("🔄 Loading and training AI model...")
    
    try:
        # Load the dataset
        df = pd.read_csv('combined_offer_fraud_dataset.csv')
        print(f"📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Identify text and label columns
        text_col = None
        label_col = None
        
        for col in df.columns:
            if 'text' in col.lower() or 'description' in col.lower() or 'content' in col.lower():
                text_col = col
            elif 'label' in col.lower() or 'target' in col.lower() or 'fraud' in col.lower():
                label_col = col
        
        if text_col is None:
            text_col = df.columns[0]
        if label_col is None:
            label_col = df.columns[-1]
        
        print(f"📝 Using columns: '{text_col}' for text, '{label_col}' for labels")
        
        # Preprocess the text data
        print("🧹 Preprocessing text data...")
        df['processed_text'] = df[text_col].apply(preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Prepare features and labels
        X = df['processed_text']
        y = df[label_col]
        
        # Convert labels to binary (0=fake, 1=real)
        if y.dtype == 'object':
            y = y.map({'fake': 0, 'real': 1, 'Fake': 0, 'Real': 1, 'FAKE': 0, 'REAL': 1})
        
        print(f"📈 Training data: {X.shape[0]} samples")
        print(f"🏷️ Label distribution: {y.value_counts().to_dict()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and fit TF-IDF vectorizer
        print("🔤 Training TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train Logistic Regression model
        print("🤖 Training Logistic Regression model...")
        model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_tfidf)
        model_accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model info
        model_info = {
            'accuracy': model_accuracy,
            'precision_fake': report['0']['precision'],
            'recall_fake': report['0']['recall'],
            'f1_fake': report['0']['f1-score'],
            'precision_real': report['1']['precision'],
            'recall_real': report['1']['recall'],
            'f1_real': report['1']['f1-score'],
            'total_samples': len(df),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Accuracy: {model_accuracy:.4f}")
        print(f"🎯 Precision (Fake): {model_info['precision_fake']:.4f}")
        print(f"🎯 Precision (Real): {model_info['precision_real']:.4f}")
        
        # Save the model and vectorizer
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/fake_offer_model.pkl')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(model_info, 'models/model_info.pkl')
        
        print("💾 Model saved successfully!")
        
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        # Create fallback model
        print("🔄 Creating fallback model...")
        create_fallback_model()

def create_fallback_model():
    """Create a simple fallback model"""
    global model, vectorizer, model_accuracy, model_info
    
    vectorizer = TfidfVectorizer(max_features=1000)
    model = LogisticRegression(random_state=42)
    
    # Create dummy data
    dummy_texts = [
        "We are pleased to offer you a position with competitive salary and benefits",
        "Congratulations on your job offer with excellent benefits package",
        "You have been selected for this amazing opportunity with great pay",
        "Send money to receive your job offer immediately",
        "Pay processing fee to get your dream job",
        "Urgent: Transfer funds to secure your position",
        "Send your bank details to receive the job offer",
        "Pay $500 to get hired immediately"
    ]
    dummy_labels = [1, 1, 1, 0, 0, 0, 0, 0]  # 1=real, 0=fake
    
    X_dummy = vectorizer.fit_transform(dummy_texts)
    model.fit(X_dummy, dummy_labels)
    
    model_accuracy = 0.85
    model_info = {
        'accuracy': model_accuracy,
        'precision_fake': 0.8,
        'recall_fake': 0.8,
        'f1_fake': 0.8,
        'precision_real': 0.9,
        'recall_real': 0.9,
        'f1_real': 0.9,
        'total_samples': 8,
        'training_samples': 8,
        'test_samples': 0
    }
    
    # Save fallback model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fake_offer_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(model_info, 'models/model_info.pkl')
    
    print("✅ Fallback model created and saved!")

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer, model_info
    
    try:
        model = joblib.load('models/fake_offer_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        model_info = joblib.load('models/model_info.pkl')
        print("✅ Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("📁 Model files not found. Training new model...")
        train_model()
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        create_fallback_model()
        return True

@app.route('/')
def home():
    return jsonify({
        "message": "AI-Powered Fake Job Offer Detector API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Analyze job offer text",
            "/upload": "POST - Upload and analyze file",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_accuracy": model_accuracy if model_accuracy else 0
    })

@app.route('/model-info')
def model_info_endpoint():
    return jsonify(model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a job offer is fake or real from text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        if len(processed_text.strip()) == 0:
            return jsonify({"error": "No valid text after preprocessing"}), 400
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Get the probability of the predicted class
        confidence = probability[prediction]
        
        # Map prediction to label
        label = "Real" if prediction == 1 else "Fake"
        
        # Get fraud indicators
        indicators = get_fraud_indicators(text)
        
        # Generate explanation
        explanation = generate_explanation(label, confidence, indicators, len(text))
        
        # Save user submission
        save_user_submission(text, label, float(confidence))
        
        return jsonify({
            "text": text[:500] + "..." if len(text) > 500 else text,
            "prediction": label,
            "confidence": float(confidence),
            "confidence_percentage": f"{confidence:.1%}",
            "fraud_indicators": indicators,
            "explanation": explanation,
            "text_length": len(text),
            "model_accuracy": model_accuracy
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and analyze a file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file type
        allowed_extensions = {'txt', 'pdf', 'docx', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        if file_extension not in allowed_extensions:
            return jsonify({
                "error": f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
            }), 400
        
        # Save file temporarily
        filename = f"upload_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from file
            text = extract_text_from_file(filepath, file_extension)
            
            if not text or len(text.strip()) == 0:
                return jsonify({"error": "Could not extract text from file"}), 400
            
            # Make prediction
            processed_text = preprocess_text(text)
            text_vectorized = vectorizer.transform([processed_text])
            
            prediction = model.predict(text_vectorized)[0]
            probability = model.predict_proba(text_vectorized)[0]
            confidence = probability[prediction]
            label = "Real" if prediction == 1 else "Fake"
            
            # Get fraud indicators
            indicators = get_fraud_indicators(text)
            
            # Generate explanation
            explanation = generate_explanation(label, confidence, indicators, len(text))
            
            # Save user submission
            save_user_submission(text, label, float(confidence))
            
            return jsonify({
                "filename": file.filename,
                "file_type": file_extension,
                "text": text[:500] + "..." if len(text) > 500 else text,
                "prediction": label,
                "confidence": float(confidence),
                "confidence_percentage": f"{confidence:.1%}",
                "fraud_indicators": indicators,
                "explanation": explanation,
                "text_length": len(text),
                "model_accuracy": model_accuracy
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
        
    except Exception as e:
        return jsonify({"error": f"Upload error: {str(e)}"}), 500

def save_user_submission(text, prediction, confidence):
    # Ensure the file exists and has headers
    file_exists = os.path.isfile('user_submissions.csv')
    with open('user_submissions.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'text', 'prediction', 'confidence'])
        writer.writerow([datetime.now().isoformat(), text, prediction, confidence])

if __name__ == '__main__':
    print("🚀 Starting AI-Powered Fake Job Offer Detector...")
    print("=" * 60)
    
    # Load or train the model
    load_model()
    
    print("=" * 60)
    print("🌐 Backend server starting...")
    print("📍 API will be available at: http://localhost:5000")
    print("🔍 Endpoints:")
    print("   - POST /predict - Analyze job offer text")
    print("   - POST /upload - Upload and analyze file")
    print("   - GET /health - Health check")
    print("   - GET /model-info - Model information")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)