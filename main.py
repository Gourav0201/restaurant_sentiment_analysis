#!/usr/bin/env python3
"""
Restaurant Review Sentiment Analysis
======================================
NLP pipeline that analyzes restaurant reviews and classifies
them as Positive, Negative, or Neutral sentiment.

Tech Stack:
- NLTK — text preprocessing, tokenization, stopwords, lemmatization
- HuggingFace Transformers — sentiment classification
- Scikit-learn — TF-IDF + Logistic Regression baseline
- FastAPI — REST API backend
- Pandas, Matplotlib, Seaborn — data analysis and visualization

Author: Gourav Yadav
"""

import re
import string
import logging
import warnings
warnings.filterwarnings("ignore")

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

from transformers import pipeline as hf_pipeline

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)


# ══════════════════════════════════════════════
# SAMPLE DATASET
# ══════════════════════════════════════════════

SAMPLE_REVIEWS = [
    {"review": "The food was absolutely amazing! Best pasta I've ever had.", "sentiment": "positive"},
    {"review": "Great service and cozy atmosphere. Will definitely come back!", "sentiment": "positive"},
    {"review": "Delicious food, friendly staff, and reasonable prices.", "sentiment": "positive"},
    {"review": "Outstanding experience from start to finish. Highly recommended!", "sentiment": "positive"},
    {"review": "The pizza was perfectly crispy with fresh toppings. Loved it!", "sentiment": "positive"},
    {"review": "Excellent quality ingredients and the chef clearly knows his craft.", "sentiment": "positive"},
    {"review": "Wonderful dining experience. The desserts were heavenly!", "sentiment": "positive"},
    {"review": "Best restaurant in town! The steak was cooked to perfection.", "sentiment": "positive"},
    {"review": "Terrible food. The pasta was overcooked and tasteless.", "sentiment": "negative"},
    {"review": "Very disappointed. Waited 45 minutes for cold food.", "sentiment": "negative"},
    {"review": "Rude staff and the portions were tiny for the price.", "sentiment": "negative"},
    {"review": "Never coming back. Found a hair in my soup.", "sentiment": "negative"},
    {"review": "Awful experience. The place was dirty and the food was stale.", "sentiment": "negative"},
    {"review": "Overpriced for what you get. Much better options nearby.", "sentiment": "negative"},
    {"review": "The worst dining experience of my life. Everything was wrong.", "sentiment": "negative"},
    {"review": "Food took forever and arrived cold. Completely unacceptable.", "sentiment": "negative"},
    {"review": "The food was okay, nothing special.", "sentiment": "neutral"},
    {"review": "Average experience. Food was decent but not memorable.", "sentiment": "neutral"},
    {"review": "It was fine. Not bad but not great either.", "sentiment": "neutral"},
    {"review": "Standard restaurant food. Service was acceptable.", "sentiment": "neutral"},
    {"review": "The meal was satisfactory. Prices are fair.", "sentiment": "neutral"},
    {"review": "Nothing to complain about but nothing to rave about either.", "sentiment": "neutral"},
]


# ══════════════════════════════════════════════
# TEXT PREPROCESSING
# ══════════════════════════════════════════════

class TextPreprocessor:
    """
    NLP text preprocessing pipeline using NLTK.
    
    Steps:
    1. Lowercase
    2. Remove special characters and numbers
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    """

    def _init_(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep sentiment words that stopwords might remove
        self.keep_words = {'not', 'no', 'never', 'very', 'too', 'most', 'more'}
        self.stop_words -= self.keep_words

    def clean_text(self, text: str) -> str:
        """Full preprocessing pipeline."""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def tokenize_and_filter(self, text: str) -> list:
        """Tokenize, remove stopwords, lemmatize."""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens
                  if t not in self.stop_words and len(t) > 2]
        return tokens

    def preprocess(self, text: str) -> str:
        """Full pipeline — returns clean string."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return ' '.join(tokens)


# ══════════════════════════════════════════════
# ML MODEL — TF-IDF + LOGISTIC REGRESSION
# ══════════════════════════════════════════════

class SentimentClassifier:
    """
    Traditional ML sentiment classifier.
    Uses TF-IDF vectorization + Logistic Regression.
    """

    def _init_(self):
        self.preprocessor = TextPreprocessor()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1
            )),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            ))
        ])
        self.is_trained = False
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def prepare_data(self, reviews: list) -> tuple:
        """Prepare dataset from reviews list."""
        df = pd.DataFrame(reviews)
        df['cleaned_review'] = df['review'].apply(self.preprocessor.preprocess)
        X = df['cleaned_review'].values
        y = df['sentiment'].values
        return X, y, df

    def train(self, reviews: list) -> dict:
        """Train the model and return evaluation metrics."""
        X, y, df = self.prepare_data(reviews)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Model trained. Accuracy: {accuracy:.2%}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        return {
            "accuracy": round(accuracy, 4),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single review."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        cleaned = self.preprocessor.preprocess(text)
        prediction = self.pipeline.predict([cleaned])[0]
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_

        prob_dict = {cls: round(float(prob), 4)
                     for cls, prob in zip(classes, probabilities)}
        confidence = round(float(max(probabilities)), 4)

        return {
            "review": text,
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": prob_dict
        }

    def visualize_results(self, reviews: list):
        """Generate sentiment distribution visualization."""
        df = pd.DataFrame(reviews)
        df['predicted'] = df['review'].apply(
            lambda x: self.predict(x)['sentiment']
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Sentiment distribution
        sentiment_counts = df['predicted'].value_counts()
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        bar_colors = [colors.get(s, '#3498db') for s in sentiment_counts.index]

        axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
        axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sentiment')
        axes[0].set_ylabel('Count')

        # Pie chart
        axes[1].pie(sentiment_counts.values,
                    labels=sentiment_counts.index,
                    colors=bar_colors,
                    autopct='%1.1f%%',
                    startangle=90)
        axes[1].set_title('Sentiment Breakdown', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Visualization saved as sentiment_distribution.png")


# ══════════════════════════════════════════════
# HUGGINGFACE TRANSFORMER MODEL
# ══════════════════════════════════════════════

class TransformerSentimentAnalyzer:
    """
    HuggingFace transformer-based sentiment analyzer.
    Uses distilbert-base-uncased-finetuned-sst-2-english (free, no API key).
    Maps binary positive/negative to three classes.
    """

    def _init_(self):
        logger.info("Loading HuggingFace sentiment model...")
        self.analyzer = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )
        logger.info("HuggingFace model loaded.")

    def predict(self, text: str) -> dict:
        """Predict sentiment using transformer model."""
        result = self.analyzer(text)[0]
        label = result['label'].lower()
        score = round(result['score'], 4)

        # Map to three classes based on confidence
        if label == 'positive' and score > 0.85:
            sentiment = 'positive'
        elif label == 'negative' and score > 0.85:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            "review": text,
            "sentiment": sentiment,
            "confidence": score,
            "raw_label": label,
            "model": "distilbert-transformer"
        }


# ══════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════

app = FastAPI(
    title="Restaurant Review Sentiment Analysis API",
    description="NLP-powered sentiment analysis for restaurant reviews using NLTK and HuggingFace",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
ml_classifier = SentimentClassifier()
transformer_analyzer = None  # Loaded on demand


@app.on_event("startup")
async def startup_event():
    """Train ML model on startup with sample data."""
    metrics = ml_classifier.train(SAMPLE_REVIEWS)
    logger.info(f"ML model ready. Accuracy: {metrics['accuracy']:.2%}")


# ── Request/Response Models ──

class ReviewRequest(BaseModel):
    review: str
    use_transformer: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "review": "The food was amazing and service was excellent!",
                "use_transformer": False
            }
        }

class BatchReviewRequest(BaseModel):
    reviews: list[str]


# ── Endpoints ──

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Restaurant Sentiment Analysis API",
        "endpoints": {
            "analyze": "POST /analyze",
            "batch": "POST /batch",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ml_model_ready": ml_classifier.is_trained,
        "transformer_loaded": transformer_analyzer is not None
    }

@app.post("/analyze")
async def analyze_sentiment(request: ReviewRequest):
    """
    Analyze sentiment of a single restaurant review.
    Set use_transformer=true for HuggingFace model (slower but more accurate).
    """
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review cannot be empty.")

    if request.use_transformer:
        global transformer_analyzer
        if transformer_analyzer is None:
            transformer_analyzer = TransformerSentimentAnalyzer()
        result = transformer_analyzer.predict(request.review)
    else:
        result = ml_classifier.predict(request.review)

    return result

@app.post("/batch")
async def batch_analyze(request: BatchReviewRequest):
    """Analyze sentiment for multiple reviews at once."""
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")

    results = []
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for review in request.reviews:
        if review.strip():
            result = ml_classifier.predict(review)
            results.append(result)
            sentiment_counts[result["sentiment"]] += 1

    return {
        "total_reviews": len(results),
        "sentiment_summary": sentiment_counts,
        "results": results
    }


# ── Main ──

if _name_ == "_main_":
    # Train and test locally
    print("Training sentiment classifier...")
    classifier = SentimentClassifier()
    metrics = classifier.train(SAMPLE_REVIEWS)
    print(f"Accuracy: {metrics['accuracy']:.2%}")

    # Test predictions
    test_reviews = [
        "Absolutely loved the food! Best restaurant ever.",
        "Terrible experience. Food was cold and stale.",
        "The food was okay, nothing special really."
    ]

    print("\nSample Predictions:")
    for review in test_reviews:
        result = classifier.predict(review)
        print(f"Review: {review[:50]}...")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})\n")

    # Generate visualization
    classifier.visualize_results(SAMPLE_REVIEWS)

    # Start API
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
