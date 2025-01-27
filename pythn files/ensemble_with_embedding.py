# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from transformers import BertTokenizer, BertModel
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# %%
data = pd.read_excel("updated_file.xlsx")

# %%
data.head()

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and prepare the data
def load_and_prepare_data(train_path):
    # Load training data
    df = data.copy()
    
    # Select features for training
    numerical_features = [
        'Years_Experience', 
        'resume_jd_similarity',
        'Transcript_jd_similarity',
        'resume_transcript_similarity',
        'resume_sentence_count',
        'resume_avg_word_length',
        'skill_match_count',
        'university_education_count',
        'transcript_vocab_diversity',
        'transcript_avg_sentence_length',
        'resume_jd_similarity_transformers',
        'transcript_jd_similarity_transformers',
        'transcript_resume_similarity_transformers',
        'Transcript_sentiment',
        'Resume_sentiment',
        'JobDescription_sentiment'
    ]
    
    # Prepare features and target
    X = df[numerical_features]
    
    # Encode target variable if it's not already numeric
    le = LabelEncoder()
    y = le.fit_transform(df['decision'])
    
    return X, y, numerical_features

# 2. Create and train the XGBoost model
def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train the model with evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb_model.fit(
        X_train, 
        y_train,
        eval_set=eval_set,
        verbose=True
    )
    
    return xgb_model

# 3. Create and train the ANN model
def train_ann(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with validation data
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    return model, history

# 4. Plot training metrics
def plot_training_metrics(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 5. Evaluate ensemble model
def evaluate_ensemble(xgb_model, ann_model, X_test, y_test, scaler):
    # Get predictions from both models
    X_test_scaled = scaler.transform(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    ann_pred_proba = ann_model.predict(X_test_scaled).ravel()
    
    # Combine predictions
    ensemble_pred_proba = (xgb_pred_proba + ann_pred_proba) / 2
    ensemble_predictions = (ensemble_pred_proba > 0.5).astype(int)
    
    # Print classification report
    print("\nEnsemble Model Classification Report:")
    print(classification_report(y_test, ensemble_predictions))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, ensemble_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 6. Main training process
def main():
    # Load and prepare data
    print("Loading and preparing data...")
    X, y, features = load_and_prepare_data('your_training_data.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Train ANN
    print("\nTraining Neural Network...")
    ann_model, history = train_ann(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Plot ANN training metrics
    plot_training_metrics(history)
    
    # Evaluate ensemble
    evaluate_ensemble(xgb_model, ann_model, X_test, y_test, scaler)
    
    # Save the models
    print("\nSaving models...")
    model_artifacts = {
        'xgb_model': xgb_model,
        'ann_model': ann_model,
        'scaler': scaler,
        'feature_names': features
    }
    
    with open('recruitment_ensemble_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print("Training complete! Models saved as 'recruitment_ensemble_model.pkl'")

if __name__ == "__main__":
    main()

# %%
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score

# Assuming model2 is the trained XGBoost model and model is the trained ANN model

# Get predicted probabilities for both models
y_test_pred_proba_xgb = model2.predict_proba(X_test)[:, 1]  # XGBoost probabilities
y_test_pred_nn = model.predict(X_test)[:, 0]  # ANN probabilities

# Combine predictions using weighted average
# You can experiment with different weights for XGBoost and ANN
alpha = 0.6  # Weight for XGBoost model
beta = 0.4   # Weight for ANN model

# Weighted average of probabilities
mean_prob = alpha * y_test_pred_proba_xgb + beta * y_test_pred_nn

# Convert to binary class predictions
ensemble_predictions = (mean_prob > 0.5).astype(int)

# Evaluate performance of ensemble model
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_predictions))
print("Ensemble ROC AUC:", roc_auc_score(y_test, mean_prob))

# Save the ensemble model (weights + models) using pickle
ensemble_model = {
    'xgb_model': model2,
    'ann_model': model,
    'alpha': alpha,
    'beta': beta
}

with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

print("Ensemble model saved as 'ensemble_model.pkl'")


# %%
with open('ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

# Access individual models and weights
xgb_model = ensemble_model['xgb_model']
ann_model = ensemble_model['ann_model']
alpha = ensemble_model['alpha']
beta = ensemble_model['beta']



