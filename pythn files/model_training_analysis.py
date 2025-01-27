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

# %%
!pip install autocorrect

# %%
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from autocorrect import Speller


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.spell_checker = Speller(lang='en')
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Cleans the input text by lowercasing, removing special characters, tokenizing, removing stopwords,
        lemmatizing, and optionally stemming.
        """
        if pd.isna(text):
            return ""

        # Lowercase the text
        text = str(text).lower()

        # Remove special characters, numbers, and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords and apply lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

        # Optionally apply stemming (comment out if not needed)
        # tokens = [self.stemmer.stem(token) for token in tokens]

        # Apply spelling correction
        tokens = [self.spell_checker(token) for token in tokens]

        return ' '.join(tokens)

    def preprocess_pipeline(self, text, correct_spelling=False, use_stemming=False):
        """
        Full preprocessing pipeline with optional spelling correction and stemming.
        """
        if pd.isna(text):
            return ""

        # Clean the text
        text = self.clean_text(text)

        # Optional: Correct spelling
        if correct_spelling:
            text = ' '.join([self.spell_checker(word) for word in text.split()])

        # Optional: Apply stemming
        if use_stemming:
            text = ' '.join([self.stemmer.stem(word) for word in text.split()])

        return text


# %% [markdown]
# ### Define Helper Classes

# %%
!pip install textstat

# %%
# Install required packages
!pip install textblob gensim textstat

# Import required libraries
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import textstat

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor with required tools and models."""
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd = TruncatedSVD(n_components=50)
        self.scaler = StandardScaler()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Preprocess text by tokenizing, lemmatizing, and removing stopwords."""
        tokens = word_tokenize(str(text).lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def extract_features(self, resume, jd, transcript):
        """Extract features from resume, job description, and transcript."""
        # Handle NaN values
        resume = str(resume) if pd.notna(resume) else ""
        jd = str(jd) if pd.notna(jd) else ""
        transcript = str(transcript) if pd.notna(transcript) else ""

        # Preprocess texts
        resume_processed = self.preprocess_text(resume)
        jd_processed = self.preprocess_text(jd)
        transcript_processed = self.preprocess_text(transcript)

        features = {}

        try:
            # TF-IDF Similarities
            tfidf_matrix = self.tfidf.fit_transform([resume_processed, jd_processed, transcript_processed])
            features['resume_jd_similarity'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
            features['resume_transcript_similarity'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])[0, 0]
            features['jd_transcript_similarity'] = cosine_similarity(tfidf_matrix[1:2], tfidf_matrix[2:3])[0, 0]

            # Sentiment Analysis (VADER)
            for text, prefix in [(resume, 'resume'), (jd, 'jd'), (transcript, 'transcript')]:
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                features.update({
                    f'{prefix}_sentiment_pos': sentiment['pos'],
                    f'{prefix}_sentiment_neg': sentiment['neg'],
                    f'{prefix}_sentiment_neu': sentiment['neu'],
                    f'{prefix}_sentiment_compound': sentiment['compound']
                })

            # Text Statistics
            for text, prefix in [(resume, 'resume'), (jd, 'jd'), (transcript, 'transcript')]:
                features[f'{prefix}_length'] = len(text.split())
                features[f'{prefix}_char_length'] = len(text)
                features[f'{prefix}_avg_word_length'] = sum(len(word) for word in text.split()) / max(len(text.split()), 1)
                features[f'{prefix}_sentence_count'] = len(text.split('.'))

            # Readability Metrics
            for text, prefix in [(resume, 'resume'), (jd, 'jd')]:
                try:
                    features[f'{prefix}_readability'] = textstat.flesch_reading_ease(text)
                    features[f'{prefix}_gunning_fog'] = textstat.gunning_fog(text)
                    features[f'{prefix}_smog'] = textstat.smog_index(text)
                    features[f'{prefix}_automated_readability'] = textstat.automated_readability_index(text)
                    features[f'{prefix}_coleman_liau'] = textstat.coleman_liau_index(text)
                except:
                    features[f'{prefix}_readability'] = 0
                    features[f'{prefix}_gunning_fog'] = 0
                    features[f'{prefix}_smog'] = 0
                    features[f'{prefix}_automated_readability'] = 0
                    features[f'{prefix}_coleman_liau'] = 0

            # Skills Matching
            technical_skills = [
                'python', 'java', 'javascript', 'c++', 'sql', 'machine learning',
                'data analysis', 'deep learning', 'nlp', 'cloud computing',
                'aws', 'azure', 'docker', 'kubernetes', 'git', 'agile'
            ]
            soft_skills = [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'analytical', 'creative', 'organized', 'management'
            ]

            features['technical_skills_resume'] = sum(skill in resume.lower() for skill in technical_skills)
            features['technical_skills_jd'] = sum(skill in jd.lower() for skill in technical_skills)
            features['soft_skills_resume'] = sum(skill in resume.lower() for skill in soft_skills)
            features['soft_skills_jd'] = sum(skill in jd.lower() for skill in soft_skills)
            features['skills_match_ratio'] = (features['technical_skills_resume'] + features['soft_skills_resume']) / \
                                             max((features['technical_skills_jd'] + features['soft_skills_jd']), 1)

            # Lexical Diversity
            for text, prefix in [(resume_processed, 'resume'), (jd_processed, 'jd'), (transcript_processed, 'transcript')]:
                tokens = text.split()
                features[f'{prefix}_lexical_diversity'] = len(set(tokens)) / max(len(tokens), 1)
                features[f'{prefix}_unique_words'] = len(set(tokens))

            # Topic Modeling with LDA
            if resume_processed and jd_processed and transcript_processed:
                dictionary = Dictionary([resume_processed.split(), jd_processed.split(), transcript_processed.split()])
                corpus = [dictionary.doc2bow(doc.split()) for doc in [resume_processed, jd_processed, transcript_processed]]
                try:
                    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)
                    resume_topics = dict(lda_model[corpus[0]])
                    jd_topics = dict(lda_model[corpus[1]])
                    for i in range(3):
                        features[f'resume_topic_{i}'] = resume_topics.get(i, 0.0)
                        features[f'jd_topic_{i}'] = jd_topics.get(i, 0.0)
                except:
                    for i in range(3):
                        features[f'resume_topic_{i}'] = 0.0
                        features[f'jd_topic_{i}'] = 0.0

            # Education and Experience Indicators
            education_terms = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
            experience_terms = ['year', 'years', 'experience', 'worked', 'work']
            features['education_mentions'] = sum(term in resume.lower() for term in education_terms)
            features['experience_mentions'] = sum(term in resume.lower() for term in experience_terms)

            # Named Entity Recognition
            try:
                resume_blob = TextBlob(resume)
                jd_blob = TextBlob(jd)
                features['resume_proper_nouns'] = len([word for word, tag in resume_blob.tags if tag == 'NNP'])
                features['jd_proper_nouns'] = len([word for word, tag in jd_blob.tags if tag == 'NNP'])
            except:
                features['resume_proper_nouns'] = 0
                features['jd_proper_nouns'] = 0

            # SVD-based dimensional reduction of TF-IDF
            svd_features = self.svd.fit_transform(tfidf_matrix)
            for i in range(min(10, svd_features.shape[1])):
                features[f'svd_component_{i}'] = svd_features[0, i]

        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            return {}

        return features

# Function to load and process data
def process_data(file_path):
    print("Starting data processing...")

    # Initialize feature extractor
    feature_extractor = FeatureExtractor()

    try:
        # Load the dataset
        data = pd.read_excel(file_path)
        print(f"Loaded dataset with {len(data)} rows")

        # Extract features
        features = []
        for idx, row in data.iterrows():
            try:
                print(f"Processing row {idx + 1}/{len(data)}", end='\r')
                feature_dict = feature_extractor.extract_features(
                    row['Resume'],
                    row['Job Description'],
                    row['Transcript']
                )
                features.append(feature_dict)
            except Exception as e:
                print(f"\nError processing row {idx}: {str(e)}")
                features.append({})

        # Convert to DataFrame and return features
        X = pd.DataFrame(features)

        print("\nFeature extraction completed.")
        print(f"Feature matrix shape: {X.shape}")

        return X

    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        return None

# Example usage
file_path = '/content/dataset_1_2_3_combined (1) (1).xlsx'
X = process_data(file_path)



# %%
## Example usage
file_path = '/content/dataset_1_2_3_combined (1) (1).xlsx'
X = process_data(file_path)

# Access the 'decision' column from the original data for labels
y = data['decision']  # Assuming 'decision' is the target column

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %% [markdown]
# # Train-Test Split

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
from sklearn.pipeline import Pipeline # Import Pipeline class from sklearn.pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(random_state=42))
])

# %% [markdown]
# # Hyperparameter Tuning and Model Training

# %%
!pip install scikit-learn==1.2.2

# %%
!pip install --upgrade xgboost

# %%
!pip install catboost

# %%
# Import required library for CatBoost
from catboost import CatBoostClassifier

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
    ),
    'CatBoost': (
        CatBoostClassifier(random_state=42, verbose=0),
        {'iterations': [100, 200, 300], 'depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
    )
}

# Train and tune models
best_models = {}
for name, (model, params) in models.items():
    print(f"Tuning {name}...")
    grid_search = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)

    # Handle XGBoost and CatBoost separately for label encoding
    if name in ['XGBoost', 'CatBoost']:
        # Create a LabelEncoder object
        le = LabelEncoder()

        # Fit the encoder to your training labels and transform them
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)  # Transform y_test using the same encoder

        # Use encoded labels for training XGBoost and CatBoost
        grid_search.fit(X_train_scaled, y_train_encoded)
    else:
        # For other models, use original y_train
        grid_search.fit(X_train_scaled, y_train)

    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")


# %% [markdown]
# # Model Evaluation

# %%
# Evaluate models
for name, model in best_models.items():
    print(f"\nEvaluating {name}...")

    # Predict the test labels
    y_pred = model.predict(X_test_scaled)

    # If the model is XGBoost or CatBoost, inverse-transform the predictions
    if name in ['XGBoost', 'CatBoost']:
        y_pred = le.inverse_transform(y_pred)  # Use the same LabelEncoder object (le)

    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# %% [markdown]
# #  Post-Model Analysis

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

# %% [markdown]
# ## SHAP Analysis for Best Model

# %%
!pip install shap

# %%

!pip install pdp

# %%
import shap
import pdp
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for probability conversion
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# ### Step 3: Access the Trained Model and Data

# %%
# Assuming 'best_models', 'X_train_scaled', 'X_test_scaled', 'y_train', 'y_test' are available

best_xgb_model = best_models['XGBoost']  # Get the best XGBoost model

# Create a LabelEncoder object
le = LabelEncoder()

# Fit the encoder to your training labels and transform them
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)  # Transform y_test using the same encoder

# Use encoded labels for training and SHAP analysis with XGBoost
best_xgb_model.fit(X_train_scaled, y_train_encoded) # Fit the model using encoded labels

# %% [markdown]
# ## Step 4: Generate SHAP Plots

# %%
# Create the SHAP explainer
explainer = shap.Explainer(best_xgb_model, X_train_scaled)
shap_values = explainer(X_test_scaled)



# %%
# Ensure X_test is a DataFrame with proper column names
if isinstance(X_test_scaled, pd.DataFrame):
    X_test_scaled_df = X_test_scaled
else:
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# %%
# a) SHAP Beeswarm Plot
shap.summary_plot(shap_values, X_test, plot_type="dot")
plt.title("SHAP Beeswarm Plot")
plt.show()



# %% [markdown]
# ## b) SHAP Waterfall Plots (for 3 instances)

# %%
for i in [0, 50, 100]:  # Adjust indices as needed
    shap.plots.waterfall(shap_values[i])
    plt.title(f"SHAP Waterfall Plot (Instance {i})")
    plt.show()

# %% [markdown]
# # c) SHAP Dependence Plots (for 3 features)

# %%
for feature in ["resume_jd_similarity", "resume_sentiment_compound", "skills_match_ratio"]:  # Replace with actual feature names
    shap.dependence_plot(feature, shap_values.values, X_test, interaction_index=None)
    plt.title(f"SHAP Dependence Plot for {feature}")
    plt.show()

# %% [markdown]
# # Step 5: Generate Partial Dependence Plots (PDPs)

# %%
!pip install pdpbox

# %%
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import matplotlib.pyplot as plt

# Ensure X_test is a DataFrame with proper column names
if not isinstance(X_test, pd.DataFrame):
    X_test_df = pd.DataFrame(X_test, columns=X_train.columns)  # Assuming X_train has correct column names
else:
    X_test_df = X_test

# List of features for PDP
selected_features = ["resume_jd_similarity", "resume_sentiment_compound", "skills_match_ratio"]  # Replace with actual feature names
available_features = [feature for feature in selected_features if feature in X_test_df.columns]

if not available_features:
    raise ValueError("None of the selected features are found in the dataset. Please verify feature names.")

# Generate PDP for each feature
for feature in available_features:
    try:
        feature_index = X_test_df.columns.get_loc(feature)  # Get feature index
        disp = PartialDependenceDisplay.from_estimator(
            best_xgb_model,
            X_test_df,
            features=[feature_index],
            feature_names=X_test_df.columns,
            grid_resolution=100,
        )
        plt.title(f"Partial Dependence Plot for {feature}")
        plt.show()
    except Exception as e:
        print(f"Error generating PDP for feature '{feature}': {e}")


# %% [markdown]
# Convert Log Odds to Probability

# %%
def log_odds_to_probability(log_odds):
    return np.exp(log_odds) / (1 + np.exp(log_odds))

# Apply to SHAP values or PDP outputs (example for SHAP values):
probabilities = log_odds_to_probability(shap_values.values)


