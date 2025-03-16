import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pastikan stopwords tersedia
nltk.download('stopwords')

##############################################
#            DATA LOADING FUNCTION           #
##############################################
def load_data(filepath):
    """Load data from CSV file"""
    data = pd.read_csv(filepath)
    return data

##############################################
#         TEXT PREPROCESSING FUNCTIONS       #
##############################################
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F" 
                               u"\U0001F300-\U0001F5FF" 
                               u"\U0001F680-\U0001F6FF" 
                               u"\U0001F700-\U0001F77F" 
                               u"\U0001F780-\U0001F7FF" 
                               u"\U0001F800-\U0001F8FF" 
                               u"\U0001F900-\U0001F9FF" 
                               u"\U0001FA00-\U0001FA6F" 
                               u"\U0001FA70-\U0001FAFF" 
                               u"\U0001F004-\U0001F0CF" 
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_numbers(text):
    return re.sub(r'\d', '', text)

def remove_username(text):
    return re.sub(r'@[^\s]+', '', text)

def case_folding(text):
    return text.lower()

def text_preprocessing(text):
    """
    Complete basic text preprocessing:
    - Remove username, URL, HTML, emoji, symbols, numbers, and perform case folding.
    """
    if not isinstance(text, str):
        return ''
    text = remove_username(text)
    text = remove_URL(text)
    text = remove_html(text)  # Termasuk penghapusan HTML
    text = remove_emoji(text)
    text = remove_symbols(text)
    text = remove_numbers(text)
    text = case_folding(text)
    return text

##############################################
#         NORMALIZATION & TOKENIZATION       #
##############################################
def clean_word(word):
    """Remove non-alphanumeric characters from a word."""
    return ''.join(filter(str.isalnum, word))

def load_normalization_dict(kamus_path):
    """
    Load normalization dictionary from a CSV file (comma-separated).
    File diharapkan memiliki dua kolom: 'tidak_baku' dan 'kata_baku'.
    """
    kamus_data = pd.read_csv(kamus_path, delimiter=',')
    return dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))

def replace_taboo_words(text, kamus_tidak_baku):
    """
    Replace non-standard words in text using the normalization dictionary.
    Hanya mengembalikan teks yang telah direplace (tanpa output tambahan).
    """
    if isinstance(text, str):
        words = text.split()
        replaced_words = []
        for word in words:
            clean_w = clean_word(word)
            if clean_w in kamus_tidak_baku:
                replaced_words.append(kamus_tidak_baku[clean_w])
            else:
                replaced_words.append(word)
        return ' '.join(replaced_words)
    return ''

def tokenize(text):
    """Tokenize the input text into words."""
    return text.split()

def remove_stopwords(tokens, language='indonesian'):
    """Remove stopwords from a list of tokens."""
    stop_words = set(stopwords.words(language))
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens):
    """Stem tokens using Sastrawi Stemmer and return as a joined string."""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return ' '.join([stemmer.stem(word) for word in tokens])

def full_text_pipeline(text, kamus_tidak_baku):
    """
    Full text processing pipeline that:
      1. Applies basic preprocessing.
      2. Normalizes text using the provided dictionary.
      3. Tokenizes text.
      4. Removes stopwords.
      5. Applies stemming.
    Returns the final processed text.
    """
    cleaned = text_preprocessing(text)
    normalized = replace_taboo_words(cleaned, kamus_tidak_baku)
    tokens = tokenize(normalized)
    tokens_no_stop = remove_stopwords(tokens)
    stemmed = stem_text(tokens_no_stop)
    return stemmed

##############################################
#           MODEL TRAINING FUNCTIONS         #
##############################################
# Fungsi train_model untuk split 90:10 (test_size=0.2)
def train_model(data, text_column='cleaning', label_column='sentimen', test_size=0.2):
    """
    Train and evaluate the SVM classification model for split 90:10.
    Proses:
      - Hapus missing values.
      - Split data dengan test_size=0.2 (90:10) dan stratifikasi.
      - TF-IDF vectorization.
      - Tangani ketidakseimbangan kelas dengan SMOTE.
      - Optimasi hyperparameter SVM (kernel: 'linear', 'rbf') menggunakan GridSearchCV.
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

# Fungsi train_model_split_80_20 untuk split 80:20 (test_size=0.2)
def train_model_split_80_20(data, text_column='steming_data', label_column='sentimen', test_size=0.2):
    """
    Train model for split 80:20 with two stages:
      1. Baseline training (without SMOTE) using GridSearchCV with kernel ['linear', 'rbf', 'poly'].
      2. Final training after SMOTE using GridSearchCV with kernel ['linear', 'rbf'].
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
    
    # Tahap 1: Baseline training (tanpa SMOTE)
    param_grid_baseline = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }
    grid_search_baseline = GridSearchCV(SVC(), param_grid_baseline, cv=5, scoring='accuracy')
    grid_search_baseline.fit(X_train_tfidf, y_train)
    y_pred_baseline = grid_search_baseline.predict(X_test_tfidf)
    # Confusion matrix baseline (bisa digunakan untuk analisis tambahan)
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    
    # Tahap 2: Final training with SMOTE
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

# Fungsi train_model_split_70_30 untuk split 70:30 (test_size=0.3)
def train_model_split_70_30(data, text_column='steming_data', label_column='sentimen', test_size=0.3):
    """
    Train model for split 70:30 with two stages:
      1. Baseline training (without SMOTE) using GridSearchCV with kernel ['linear', 'rbf', 'poly'].
      2. Final training after SMOTE using GridSearchCV with kernel ['linear', 'rbf'].
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
    y_pred_baseline = grid_search_baseline.predict(X_test_tfidf)
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    
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

# Fungsi train_model_split_60_40 untuk split 60:40 (test_size=0.4)
def train_model_split_60_40(data, text_column='steming_data', label_column='sentimen', test_size=0.4):
    """
    Train model for split 60:40 with two stages:
      1. Baseline training (without SMOTE) using GridSearchCV with kernel ['linear', 'rbf', 'poly'].
      2. Final training after SMOTE using GridSearchCV with kernel ['linear', 'rbf'].
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
    y_pred_baseline = grid_search_baseline.predict(X_test_tfidf)
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    
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
#  (Dipertahankan sesuai kode asli utils.py) #
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

##############################################
#            END OF MODULE                   #
##############################################
