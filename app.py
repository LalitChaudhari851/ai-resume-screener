import os
import pandas as pd
import spacy
import pickle
from flask import Flask, render_template, request, session, send_file
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx

# --- Initialization ---
app = Flask(__name__)
app.secret_key = 'super-secret-key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model, vectorizer, and spaCy model
try:
    with open('model/logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    nlp = spacy.load('en_core_web_sm')
    print("Models and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer not found. Please run train_model.py first.")
    exit()

# Pre-defined list of skills for extraction
SKILLS_DB = [
    'python', 'java', 'c++', 'javascript', 'sql', 'django', 'flask',
    'react', 'angular', 'vue', 'node.js', 'mongodb', 'aws', 'azure',
    'docker', 'kubernetes', 'git', 'scikit-learn', 'tensorflow', 'pytorch',
    'data analysis', 'machine learning', 'project management', 'agile'
]

# --- Helper Functions ---
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(lemmas)

def extract_skills(text):
    doc = nlp(text.lower())
    found_skills = set()
    for token in doc:
        if token.lemma_ in SKILLS_DB:
            found_skills.add(token.lemma_)
    for skill in SKILLS_DB:
        if ' ' in skill and skill in text.lower():
            found_skills.add(skill)
    return list(found_skills)

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        uploaded_files = request.files.getlist('resumes')

        results = []
        if not uploaded_files or uploaded_files[0].filename == '':
            return render_template('index.html', error="No resume files selected.")
        
        processed_jd = preprocess_text(job_description)
        jd_vector = vectorizer.transform([processed_jd])

        for file in uploaded_files:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                resume_text = extract_text_from_docx(file_path)
            else:
                os.remove(file_path)
                continue

            processed_resume = preprocess_text(resume_text)
            resume_vector = vectorizer.transform([processed_resume])
            similarity_score = cosine_similarity(resume_vector, jd_vector)[0][0]
            prediction = model.predict(resume_vector)[0]
            predicted_fit = "Good Fit" if prediction == 1 else "Needs Review"
            skills = extract_skills(resume_text)

            results.append({
                'filename': filename,
                'similarity': f"{similarity_score * 100:.2f}%",
                'prediction': predicted_fit,
                'skills': ", ".join(skills) if skills else "No relevant skills found."
            })
            
            os.remove(file_path)
        
        results.sort(key=lambda x: float(x['similarity'][:-1]), reverse=True)
        session['results'] = results
        return render_template('results.html', results=results, job_description=job_description)
    
    return render_template('index.html')

@app.route('/download_csv')
def download_csv():
    results = session.get('results', [])
    if not results:
        return "No results to download.", 404
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'shortlisted_resumes.csv')
    df.to_csv(csv_path, index=False)
    
    return send_file(csv_path, as_attachment=True, download_name='shortlisted_candidates.csv')

if __name__ == '__main__':
    app.run(debug=True)