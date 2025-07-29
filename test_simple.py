import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("Testing basic dependencies...")

# Test pandas
try:
    df = pd.read_csv('combined_offer_fraud_dataset.csv')
    print(f"✅ Pandas OK - Dataset loaded: {df.shape}")
except Exception as e:
    print(f"❌ Pandas error: {e}")

# Test sklearn
try:
    vectorizer = TfidfVectorizer(max_features=100)
    model = LogisticRegression()
    print("✅ Scikit-learn OK")
except Exception as e:
    print(f"❌ Scikit-learn error: {e}")

# Test joblib
try:
    joblib.dump(model, 'test_model.pkl')
    loaded_model = joblib.load('test_model.pkl')
    print("✅ Joblib OK")
except Exception as e:
    print(f"❌ Joblib error: {e}")

print("Basic dependency test completed!") 