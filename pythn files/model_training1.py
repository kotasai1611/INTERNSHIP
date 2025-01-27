# %% [markdown]
# ### Recruitment Decision Analysis  
# ----------------------------  
# This script analyzes recruitment decisions based on resume, job description, and transcript data. It includes:  
# 1. **Feature Extraction**: Extracts features like resume-JD similarity, sentiment analysis, text length, and skills matching from text data.  
# 2. **Train-Test Split**: Splits data into training (80%) and testing (20%) sets using sklearn.  
# 3. **Hyperparameter Tuning**: Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost models with grid search for hyperparameter tuning.  
# 4. **Model Evaluation**: Evaluates models using metrics like accuracy, ROC AUC, and classification reports.  
# 5. **In-depth Statistical Analysis**: Performs logistic regression analysis using statsmodels for statistical insights.  
# 6. **Post-Model Analysis**: Includes feature importance visualization for the best-performing model.  

# %% [markdown]
# ### Import necessary libraries

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import spacy
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download
download('punkt')
download('stopwords')
download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


# %% [markdown]
# # Load spaCy model

# %%
nlp = spacy.load('en_core_web_sm')

# %% [markdown]
# # Load dataset

# %%
data = pd.read_excel("/content/dataset_1_2_3_combined (1).xlsx")

# %% [markdown]
# ####E.D.A

# %%
print("Dataset Info:")
print(data.info())
print("\nTarget Variable Distribution:")
print(data['decision'].value_counts())


# %%
# Plot Target Variable Distribution
sns.countplot(x='decision', data=data)
plt.title("Target Variable Distribution")
plt.show()


# %%
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# %%
# Correlation heatmap (only for numerical features)
plt.figure(figsize=(10, 8))
# Select only numerical features for correlation calculation
numerical_data = data.select_dtypes(include=np.number)
sns.heatmap(numerical_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# %% [markdown]
# # Step 2: Preprocessing
# 

# %%
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download the 'punkt_tab' resource
nltk.download('punkt_tab')

# Rest of the code remains the same

# %% [markdown]
# ### Define Helper Classes

# %%
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

class FeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.svd = TruncatedSVD(n_components=50)

    def extract_features(self, resume, jd, transcript):
        features = {}

        # TF-IDF and similarities
        tfidf_matrix = self.tfidf.fit_transform([resume, jd, transcript])
        features['resume_jd_similarity'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
        features['resume_transcript_similarity'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])[0, 0]

        # Sentiment analysis
        features['resume_sentiment_polarity'] = TextBlob(resume).sentiment.polarity
        features['jd_sentiment_polarity'] = TextBlob(jd).sentiment.polarity

        # Text statistics
        features['resume_length'] = len(resume.split())
        features['jd_length'] = len(jd.split())

        # Skills matching
        skills = ['python', 'machine learning', 'data analysis', 'sql', 'deep learning']
        resume_skills = sum(skill in resume.lower() for skill in skills)
        jd_skills = sum(skill in jd.lower() for skill in skills)
        features['skills_match_count'] = resume_skills
        features['skills_match_ratio'] = resume_skills / jd_skills if jd_skills > 0 else 0

        return features


# %%
# Upload your dataset
from google.colab import files
uploaded = files.upload()

# Load data
filename = next(iter(uploaded))
data = pd.read_excel(filename)

# Initialize preprocessor and feature extractor
preprocessor = TextPreprocessor()
feature_extractor = FeatureExtractor()

# Preprocess text
data['clean_resume'] = data['Resume'].apply(preprocessor.clean_text)
data['clean_jd'] = data['Job Description'].apply(preprocessor.clean_text)
data['clean_transcript'] = data['Transcript'].apply(preprocessor.clean_text)

# Extract features
features = []
for _, row in data.iterrows():
    features.append(feature_extractor.extract_features(row['clean_resume'], row['clean_jd'], row['clean_transcript']))

X = pd.DataFrame(features)
y = data['decision'].map({'select': 1, 'reject': 0})


# %% [markdown]
# # Train-Test Split

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %% [markdown]
# # Hyperparameter Tuning and Model Training

# %%
!pip install scikit-learn==1.2.2

# %%
!pip install --upgrade xgboost

# %%
# Define models and parameter grids
models = {
    'Logistic Regression': (
        LogisticRegression(random_state=42, max_iter=1000),
        {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    ),
    'Decision Tree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]}
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
    ),
    'XGBoost': (
        XGBClassifier(random_state=42, use_label_encoder=False),
        {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
    )
}

# Train and tune models
best_models = {}
for name, (model, params) in models.items():
    print(f"Tuning {name}...")
    grid_search = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")


# %% [markdown]
# # Model Evaluation

# %%
# Evaluate models
for name, model in best_models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# %% [markdown]
# #  Post-Model Analysis

# %%
# Logistic Regression Analysis with statsmodels
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_train_sm)
logit_results = logit_model.fit()
print("\nLogistic Regression Summary:")
print(logit_results.summary())

# %%
# Feature importance for Random Forest

# %%
if 'Random Forest' in best_models:
    rf_model = best_models['Random Forest']
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance for Random Forest:")
    print(feature_importances)
    feature_importances.head(10).plot(kind='bar', x='Feature', y='Importance', title='Top 10 Features')


