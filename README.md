# AI-Powered Resume Screening System 🤖

This project is a web-based application that automates the initial resume screening process. It uses Natural Language Processing (NLP) and Machine Learning to parse resumes, calculate a similarity score against a given job description, and rank candidates, helping recruiters to quickly identify the most promising applicants.

---

### ## Key Features ✨

- **File Upload:** Accepts multiple resumes at once (PDF and DOCX formats).
- **Job Description Analysis:** Processes a job description to extract key requirements.
- **Cosine Similarity Scoring:** Ranks each resume based on its textual similarity to the job description.
- **ML-based Prediction:** A pre-trained model predicts if a resume is a "Good Fit" based on general characteristics.
- **Automated Skill Extraction:** Identifies relevant skills from resume text.
- **Ranked Results:** Displays a clean, ranked list of all candidates.
- **CSV Export:** Allows downloading the shortlisted candidates' data for offline use.

---

### ## Tech Stack 🛠️

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn
- **NLP:** spaCy
- **Data Handling:** Pandas, PyPDF2, python-docx
- **Frontend:** HTML, Bootstrap

---

### ## Installation & Usage 🚀

Follow these steps to get the project up and running on your local machine.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/ai-resume-screener.git](https://github.com/your-username/ai-resume-screener.git)
cd ai-resume-screener
