"""
skill_taxonomy.py
-----------------
Curated skill keyword taxonomy organised by domain.
Used by the skill extractor to identify skills in resume text.
"""

SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
        "go", "golang", "rust", "kotlin", "swift", "php", "ruby", "matlab",
        "bash", "shell", "perl", "sql"
    ],
    "web_development": [
        "html", "css", "react", "reactjs", "angular", "angularjs", "vue", "vuejs",
        "nodejs", "node.js", "express", "expressjs", "django", "flask", "fastapi",
        "spring", "springboot", "spring boot", "rest", "restful", "api", "graphql",
        "bootstrap", "tailwind", "sass", "webpack", "next.js", "nextjs", "nuxt"
    ],
    "data_science_ml": [
        "machine learning", "deep learning", "neural network", "nlp",
        "natural language processing", "computer vision", "reinforcement learning",
        "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "xgboost",
        "lightgbm", "catboost", "pandas", "numpy", "scipy", "matplotlib",
        "seaborn", "plotly", "huggingface", "transformers", "bert", "gpt",
        "regression", "classification", "clustering", "feature engineering",
        "model deployment", "mlops", "model evaluation", "cross validation"
    ],
    "data_engineering": [
        "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "cassandra",
        "sqlite", "oracle", "spark", "apache spark", "hadoop", "kafka",
        "apache kafka", "airflow", "apache airflow", "dbt", "etl",
        "data pipeline", "data warehouse", "snowflake", "bigquery", "redshift",
        "databricks", "hive", "flink", "nifi"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "ci/cd", "github actions", "gitlab ci",
        "linux", "unix", "nginx", "apache", "microservices", "serverless",
        "lambda", "ec2", "s3", "cloudformation", "helm"
    ],
    "data_analytics_bi": [
        "tableau", "power bi", "looker", "superset", "metabase", "excel",
        "google sheets", "data analysis", "data visualisation", "visualization",
        "dashboard", "reporting", "kpi", "metrics", "a/b testing",
        "statistical analysis", "statistics", "hypothesis testing"
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "project management", "agile", "scrum", "kanban",
        "presentation", "stakeholder management", "mentoring", "collaboration",
        "time management", "adaptability"
    ],
    "version_control_tools": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "trello", "notion", "slack", "postman", "vs code", "jupyter",
        "anaconda", "linux"
    ]
}

# Flat list of all skills for quick lookup
ALL_SKILLS = sorted(set(
    skill for category in SKILL_TAXONOMY.values() for skill in category
))


def get_skills_by_category(category: str) -> list:
    """Return skills for a given category name."""
    return SKILL_TAXONOMY.get(category, [])


def get_all_categories() -> list:
    """Return all category names."""
    return list(SKILL_TAXONOMY.keys())
