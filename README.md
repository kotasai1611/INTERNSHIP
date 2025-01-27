# Automated Recruitment Pipeline

## Overview
This project aims to automate the recruitment pipeline using Artificial Intelligence (AI) to streamline the hiring process. The pipeline includes automated resume screening, interview generation, and decision-making, all facilitated by AI agents. The system automates the following key steps:
- Resume screening using AI
- AI-generated interview questions based on the job description (JD)
- AI or human-led interview process
- Candidate selection/rejection decision by AI
- Automated email notifications to HR/Manager about the final decision

## Project Structure

### 1. *Data Collection*
   - Resumes and associated candidate details (either provided by the instructor or manually collected).
   - Job descriptions for various roles.

### 2. *Exploratory Data Analysis (EDA)*
   - Perform data exploration using Python libraries such as Pandas, Matplotlib, and Seaborn.
   - Generate insights through plots and statistics.

### 3. *Model Training*
   - *Classical Machine Learning Models*: Use models like XGBoost, Random Forest, and Logistic Regression.
   - *Deep Learning*: Implement a neural network using BERT embeddings for feature extraction and model training.
   - Evaluate and compare model performance to determine the best-performing model.

### 4. *Post Model Analysis*
   - SHAP (SHapley Additive exPlanations) for model interpretability.
   - Analyze feature importance and the decision-making process of the model.

### 5. *Resume Screening*
   - Implement an AI-based resume screening process, using natural language processing (NLP) techniques to match resumes with job descriptions.

### 6. *Prediction & Automated Email*
   - Once a candidate's interview is completed, the system uses the trained models to predict whether the candidate is a good fit for the job.
   - Automated emails are sent to HR/Manager with the final decision.

### 7. *Conclusion*
   - The project flowchart demonstrates the entire automated recruitment process:
     1. Collect resumes
     2. Screen resumes (AI)
     3. Select candidates for interviews (AI)
     4. Generate interview questions using the job description (AI with Llama)
     5. Conduct the interview (AI or human)
     6. Make decision (Reject/Select) (AI)
     7. Send email to the concerned person (AI)

## Dependencies
- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TensorFlow or PyTorch (for Deep Learning)
- Hugging Face Transformers (for BERT embeddings)
- SHAP
- SMTP library (for email functionality)

