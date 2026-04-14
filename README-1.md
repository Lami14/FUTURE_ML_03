# 🔍 Resume Screening System — Future Interns ML Task 3 (2026)

An ML-powered resume screening and candidate ranking system built with NLP techniques.

## 📌 Overview

This system automatically screens, scores, and ranks resumes against a given job description using:
- **Text Preprocessing** — cleaning, tokenization, lemmatization
- **Skill Extraction** — NLP-based keyword matching against a skill taxonomy
- **TF-IDF Vectorization + Cosine Similarity** — semantic resume-to-JD matching
- **Candidate Ranking** — composite scoring with skill gap analysis
- **Interactive Dashboard** — Streamlit UI for recruiters

## 🗂️ Project Structure

```
FUTURE_ML_03/
├── data/
│   └── resume_dataset.csv        # Kaggle dataset (snehaanbhawal)
├── notebooks/
│   └── resume_screening.ipynb    # Full EDA + model notebook
├── app/
│   └── streamlit_app.py          # Recruiter-facing dashboard
├── src/
│   ├── preprocessor.py           # Text cleaning pipeline
│   ├── skill_extractor.py        # NLP skill extraction
│   ├── scorer.py                 # Similarity scoring & ranking
│   └── skill_taxonomy.py         # Curated skill keyword list
├── requirements.txt
└── README.md
```

## 🚀 Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Lami14/FUTURE_ML_03.git
cd FUTURE_ML_03

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Add dataset
# Download from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
# Place resume_dataset.csv in the data/ folder

# 5. Run the dashboard
streamlit run app/streamlit_app.py
```

## 📊 How Scoring Works

Each resume receives a **composite score (0–100)** based on:

| Component | Weight | Description |
|-----------|--------|-------------|
| Skill Match | 50% | % of required skills found in resume |
| TF-IDF Similarity | 35% | Cosine similarity to job description |
| Experience Bonus | 15% | Years/seniority signals in text |

Candidates are then ranked highest → lowest with a full skill gap report.

## 🛠️ Tech Stack

- Python 3.10+
- spaCy `en_core_web_sm`
- NLTK
- scikit-learn
- pandas / numpy
- Streamlit
- matplotlib / plotly

## 📁 Dataset

**Resume Dataset** by Sneha Anbhawal (Kaggle)  
🔗 https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Contains 2,484 resumes across 25 job categories in CSV format.

## 👤 Author

**Lamla Mhlana**  
Future Interns — ML Internship Track, Task 3 (2026)  
GitHub: [@Lami14](https://github.com/Lami14)
