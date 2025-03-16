import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

##############################################
#            DATA LOADING FUNCTION           #
##############################################
def load_data(filepath):
    """Load data from CSV file"""
    data = pd.read_csv(filepath)
    return data

##############################################
#         MODEL TRAINING FUNCTIONS         #
##############################################
def train_model(data, text_column='text', label_column='sentimen', test_size=0.2):
    """
    Train and evaluate the SVM classification model.
    Proses:
      - Menghapus missing values.
      - Split data dengan test_size=0.2 dan stratifikasi.
      - TF-IDF vectorization.
      - Tangani ketidakseimbangan kelas dengan SMOTE.
      - Optimasi hyperparameter SVM menggunakan GridSearchCV.
    Mengembalikan: accuracy, classification report (dictionary), y_test, y_pred.
    """
    data = data.dropna(subset=[text_column, label_column])
    X = data[text_column]
    y = data[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    y_pred = grid_search.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report, y_test, y_pred

def train_model_split_80_20(data, text_column='text', label_column='sentimen', test_size=0.2):
    """
    Train model for split 80:20.
    Mengembalikan: accuracy, classification report, y_test, y_pred.
    """
    data = data.dropna(subset=[text_column, label_column])
    X = data[text_column]
    y = data[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Tahap 1: Baseline training tanpa SMOTE
    param_grid_baseline = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }
    grid_search_baseline = GridSearchCV(SVC(), param_grid_baseline, cv=5, scoring='accuracy')
    grid_search_baseline.fit(X_train_tfidf, y_train)
    
    # Tahap 2: Final training dengan SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    
    param_grid_resampled = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    grid_search_resampled = GridSearchCV(SVC(), param_grid_resampled, cv=5, scoring='accuracy')
    grid_search_resampled.fit(X_train_res, y_train_res)
    
    y_pred = grid_search_resampled.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report, y_test, y_pred

def train_model_split_70_30(data, text_column='text', label_column='sentimen', test_size=0.3):
    """
    Train model for split 70:30.
    Mengembalikan: accuracy, classification report, y_test, y_pred.
    """
    data = data.dropna(subset=[text_column, label_column])
    X = data[text_column]
    y = data[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    param_grid_baseline = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }
    grid_search_baseline = GridSearchCV(SVC(), param_grid_baseline, cv=5, scoring='accuracy')
    grid_search_baseline.fit(X_train_tfidf, y_train)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    
    param_grid_resampled = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    grid_search_resampled = GridSearchCV(SVC(), param_grid_resampled, cv=5, scoring='accuracy')
    grid_search_resampled.fit(X_train_res, y_train_res)
    
    y_pred = grid_search_resampled.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report, y_test, y_pred

def train_model_split_60_40(data, text_column='text', label_column='sentimen', test_size=0.4):
    """
    Train model for split 60:40.
    Mengembalikan: accuracy, classification report, y_test, y_pred.
    """
    data = data.dropna(subset=[text_column, label_column])
    X = data[text_column]
    y = data[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    param_grid_baseline = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }
    grid_search_baseline = GridSearchCV(SVC(), param_grid_baseline, cv=5, scoring='accuracy')
    grid_search_baseline.fit(X_train_tfidf, y_train)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    
    param_grid_resampled = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    grid_search_resampled = GridSearchCV(SVC(), param_grid_resampled, cv=5, scoring='accuracy')
    grid_search_resampled.fit(X_train_res, y_train_res)
    
    y_pred = grid_search_resampled.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report, y_test, y_pred

##############################################
#         VISUALIZATION FUNCTIONS            #
##############################################
def plot_sentiment_distribution(sentiment_counts, figsize=(8, 5)):
    sentiment_df = sentiment_counts.reset_index()
    sentiment_df.columns = ['sentimen', 'count']
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='sentimen', y='count', data=sentiment_df)
    ax.set_title("Distribusi Sentimen", fontsize=14)
    ax.set_xlabel("Class Sentiment", fontsize=12)
    ax.set_ylabel("Jumlah", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    return fig

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Negatif', 'Positif'], columns=['Negatif', 'Positif'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def plot_classification_metrics(report):
    classes = ['Negatif', 'Positif']
    metrics = {
        'Precision': [report[cls]['precision'] for cls in classes],
        'Recall': [report[cls]['recall'] for cls in classes],
        'F1-score': [report[cls]['f1-score'] for cls in classes]
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + (i * width), values, width, label=metric_name)
    ax.set_title('Classification Metrics per Class', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.1)
    return fig
