import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

print("--- Training Script Started ---")

# --- 1. Load and Prepare Data ---
try:
    data = pd.read_csv('sample_resumes.csv')
    print(f"Dataset loaded successfully. Shape: {data.shape}")
except FileNotFoundError:
    print("Error: 'sample_resumes.csv' not found. Please create it.")
    exit()

X = data['text']
y = data['label']

# --- 2. TF-IDF Vectorization ---
print("Vectorizing text data with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)
print("Vectorization complete.")

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# --- 4. Train a Logistic Regression Model ---
print("Training a Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate the Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# --- 6. Save the Model and Vectorizer ---
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

with open(os.path.join(model_dir, 'logistic_regression_model.pkl'), 'wb') as f:
    pickle.dump(model, f)
print("Model saved to 'model/logistic_regression_model.pkl'")

with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)
print("Vectorizer saved to 'model/tfidf_vectorizer.pkl'")

print("--- Training Script Finished ---")