"""
streamlit_app.py
----------------
Recruiter-facing dashboard for the Resume Screening System.
Run with: streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from src.preprocessor import basic_clean
from src.skill_extractor import get_flat_skill_set, compute_skill_gap
from src.scorer import rank_candidates, get_top_n, score_summary

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, professional dark-accent theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background: #0f1117; }

.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #4fc3f7;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #8b92a5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.skill-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
}
.skill-match  { background: #1b3a2a; color: #4caf80; border: 1px solid #2d6b47; }
.skill-miss   { background: #3a1b1b; color: #f47c7c; border: 1px solid #6b2d2d; }
.skill-extra  { background: #1b2a3a; color: #7cb8f4; border: 1px solid #2d4a6b; }

.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 3px solid #4fc3f7;
    padding-left: 12px;
    margin: 24px 0 12px;
}

.rank-badge {
    background: linear-gradient(135deg, #4fc3f7, #0284c7);
    color: white;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 Resume Screener")
    st.markdown("---")

    st.markdown("### 1. Upload Resumes")
    uploaded_file = st.file_uploader(
        "Upload resume_dataset.csv",
        type=["csv"],
        help="Download from: kaggle.com/datasets/snehaanbhawal/resume-dataset"
    )

    st.markdown("### 2. Job Description")
    default_jd = """We are looking for a Data Engineer to join our analytics team.

Requirements:
- 2+ years of experience in data engineering
- Strong Python and SQL skills
- Experience with Apache Airflow, dbt, or similar tools
- Familiarity with AWS, GCP, or Azure
- PostgreSQL, BigQuery, or Snowflake experience
- Knowledge of Apache Spark or Kafka is a plus
- Ability to build ETL/ELT pipelines
- Understanding of data warehousing
- Docker and Linux/Unix experience
- Good communication and teamwork skills"""

    jd_text = st.text_area(
        "Paste job description",
        value=default_jd,
        height=260
    )

    st.markdown("### 3. Settings")
    max_candidates = st.slider("Max candidates to screen", 20, 500, 100, step=20)
    top_n = st.slider("Top N to display", 5, 30, 10)
    shortlist_threshold = st.slider("Shortlist threshold (score)", 30, 90, 60)

    run_screening = st.button("▶ Run Screening", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("Future Interns — ML Task 3 (2026)")
    st.caption("Built by Lamla Mhlana")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("# 🔍 Resume Screening System")
st.markdown("AI-powered candidate screening and ranking for recruiters.")
st.markdown("---")

if not uploaded_file:
    st.info("👈 Upload your **resume_dataset.csv** in the sidebar to get started.")
    with st.expander("ℹ️  How scoring works"):
        st.markdown("""
| Component | Weight | Description |
|-----------|--------|-------------|
| **Skill Match** | 50% | % of required JD skills found in resume |
| **TF-IDF Similarity** | 35% | Cosine similarity between resume and JD vectors |
| **Experience Bonus** | 15% | Years of experience signals in resume text |

**Shortlist threshold:** Candidates with a composite score ≥ threshold are flagged for review.
        """)
    st.stop()

# Load data
@st.cache_data(show_spinner="Loading dataset...")
def load_data(file, n):
    df = pd.read_csv(file)
    df = df.rename(columns={'Resume_str': 'resume_text', 'Category': 'category'})
    df = df.dropna(subset=['resume_text'])
    df['resume_text'] = df['resume_text'].astype(str)
    df['candidate_name'] = ['Candidate_' + str(i+1).zfill(4) for i in range(len(df))]
    return df[['candidate_name', 'category', 'resume_text']].head(n)

df = load_data(uploaded_file, max_candidates)
st.success(f"✅ Loaded **{len(df)} resumes** across **{df['category'].nunique()} categories**")

# Run screening
if run_screening or 'ranked_df' not in st.session_state:
    with st.spinner(f"Screening {len(df)} candidates... this may take a moment."):
        ranked_df = rank_candidates(df, jd_text)
        st.session_state['ranked_df'] = ranked_df
        st.session_state['jd_text'] = jd_text

ranked_df = st.session_state.get('ranked_df', None)

if ranked_df is None:
    st.warning("Click **▶ Run Screening** to start.")
    st.stop()

summary = score_summary(ranked_df)
shortlisted = ranked_df[ranked_df['composite_score'] >= shortlist_threshold]

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (col1, len(ranked_df), "Total Screened"),
    (col2, len(shortlisted), "Shortlisted"),
    (col3, f"{summary['avg_composite']}", "Avg Score"),
    (col4, f"{summary['max_composite']}", "Top Score"),
    (col5, f"{round(len(shortlisted)/len(ranked_df)*100,1)}%", "Shortlist Rate"),
]
for col, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🏆 Rankings", "📊 Analytics", "🔎 Candidate Detail"])

# --- Tab 1: Rankings ---
with tab1:
    st.markdown(f'<div class="section-header">Top {top_n} Candidates</div>', unsafe_allow_html=True)

    top10 = get_top_n(ranked_df, top_n)

    # Horizontal bar chart
    fig = px.bar(
        top10[::-1],
        x="composite_score",
        y="candidate_name",
        orientation="h",
        color="composite_score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        text="composite_score",
        labels={"composite_score": "Composite Score", "candidate_name": ""},
        title=f"Top {top_n} Ranked Candidates"
    )
    fig.add_vline(x=shortlist_threshold, line_dash="dash", line_color="rgba(255,100,100,0.7)",
                  annotation_text=f"Shortlist ≥{shortlist_threshold}")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#e2e8f0", coloraxis_showscale=False,
        height=420, margin=dict(l=10, r=60, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Full Ranked Table</div>', unsafe_allow_html=True)

    display_df = ranked_df[[
        'rank', 'candidate_name', 'category',
        'composite_score', 'skill_score', 'tfidf_score', 'experience_score'
    ]].copy()
    display_df.columns = ['Rank', 'Candidate', 'Category', 'Composite', 'Skill Match', 'TF-IDF', 'Experience']

    st.dataframe(
        display_df.style
            .background_gradient(subset=['Composite'], cmap='RdYlGn', vmin=0, vmax=100)
            .format({'Composite': '{:.1f}', 'Skill Match': '{:.1f}',
                     'TF-IDF': '{:.1f}', 'Experience': '{:.1f}'}),
        use_container_width=True,
        height=400
    )

    csv = ranked_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Full Results CSV", csv, "ranked_candidates.csv", "text/csv")


# --- Tab 2: Analytics ---
with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        # Score distribution
        fig2 = px.histogram(
            ranked_df, x="composite_score", nbins=20,
            title="Composite Score Distribution",
            labels={"composite_score": "Score"},
            color_discrete_sequence=["#4fc3f7"]
        )
        fig2.add_vline(x=shortlist_threshold, line_dash="dash", line_color="red",
                       annotation_text="Threshold")
        fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                           font_color="#e2e8f0", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        # Score components radar for top candidate
        top1 = ranked_df.iloc[0]
        fig3 = go.Figure(go.Scatterpolar(
            r=[top1['skill_score'], top1['tfidf_score'], top1['experience_score'],
               top1['skill_score']],
            theta=['Skill Match', 'TF-IDF Sim', 'Experience', 'Skill Match'],
            fill='toself',
            line_color='#4fc3f7',
            fillcolor='rgba(79, 195, 247, 0.2)'
        ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"Top Candidate Profile: {top1['candidate_name']}",
            paper_bgcolor="#0f1117", font_color="#e2e8f0"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Most missing skills
    all_missing = []
    for _, row in ranked_df.iterrows():
        all_missing.extend(row['missing_skills'])

    if all_missing:
        miss_freq = pd.DataFrame(Counter(all_missing).most_common(15),
                                 columns=['skill', 'count'])
        fig4 = px.bar(
            miss_freq, x='skill', y='count',
            title='Most Commonly Missing Skills (across all candidates)',
            labels={'count': 'Candidates Missing', 'skill': ''},
            color='count', color_continuous_scale='Reds'
        )
        fig4.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                           font_color="#e2e8f0", xaxis_tickangle=-35,
                           coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Category breakdown
    cat_scores = ranked_df.groupby('category')['composite_score'].mean().sort_values(ascending=False)
    fig5 = px.bar(
        x=cat_scores.index, y=cat_scores.values,
        title='Average Score by Resume Category',
        labels={'x': 'Category', 'y': 'Avg Composite Score'},
        color=cat_scores.values, color_continuous_scale='Blues'
    )
    fig5.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                       font_color="#e2e8f0", xaxis_tickangle=-35,
                       coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)


# --- Tab 3: Candidate Detail ---
with tab3:
    st.markdown('<div class="section-header">Individual Candidate Report</div>', unsafe_allow_html=True)

    candidate_options = ranked_df['candidate_name'].tolist()
    selected = st.selectbox("Select a candidate", candidate_options)

    row = ranked_df[ranked_df['candidate_name'] == selected].iloc[0]

    col_a, col_b, col_c, col_d = st.columns(4)
    for col, val, label in [
        (col_a, f"{row['composite_score']:.1f}", "Composite"),
        (col_b, f"{row['skill_score']:.1f}", "Skill Match"),
        (col_c, f"{row['tfidf_score']:.1f}", "TF-IDF"),
        (col_d, f"{row['experience_score']:.1f}", "Experience"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_skills, col_resume = st.columns([1, 2])

    with col_skills:
        st.markdown('<div class="section-header">Skill Analysis</div>', unsafe_allow_html=True)

        st.markdown("**✅ Matched Skills**")
        if row['matched_skills']:
            tags = " ".join([f'<span class="skill-tag skill-match">{s}</span>' for s in row['matched_skills']])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.caption("No matched skills found.")

        st.markdown("**❌ Missing Skills**")
        if row['missing_skills']:
            tags = " ".join([f'<span class="skill-tag skill-miss">{s}</span>' for s in row['missing_skills']])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.success("No missing skills — full match!")

        st.markdown("**➕ Additional Skills**")
        extra = row['extra_skills'][:15]
        if extra:
            tags = " ".join([f'<span class="skill-tag skill-extra">{s}</span>' for s in extra])
            st.markdown(tags, unsafe_allow_html=True)

    with col_resume:
        st.markdown('<div class="section-header">Resume Text (Preview)</div>', unsafe_allow_html=True)
        preview = row['resume_text'][:1500] + "..." if len(row['resume_text']) > 1500 else row['resume_text']
        st.text_area("", value=preview, height=320, disabled=True)
