# %%
import re
import spacy
import contractions
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import nltk

# %%
data = pd.read_excel('updated_file.xlsx')

# %%
# Use a single TfidfVectorizer for both Resume and Job Description
tfidf_vectorizer4 = TfidfVectorizer()
combined_text = data['Resume_processed'] + " " + data['Job_Description_processed']
tfidf_vectorizer4.fit(combined_text)

# Split the combined TF-IDF matrix back into Resume and Job Description matrices
resume_matrix = tfidf_vectorizer4.transform(data['Resume_processed'])
jobDescription_matrix = tfidf_vectorizer4.transform(data['Job_Description_processed'])

# Calculate row-wise cosine similarity
cos_rjd = []
for i in range(data.shape[0]):
    cos_rjd.append(cosine_similarity(resume_matrix[i], jobDescription_matrix[i])[0][0])

# %%
data['resume_jd_simi'] = cos_rjd
data.groupby(['Role' , 'decision'])['resume_jd_simi'].mean()

# %%



