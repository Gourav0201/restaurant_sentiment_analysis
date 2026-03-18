#!/usr/bin/env python3
"""
Restaurant Review Sentiment Analysis
======================================
NLP pipeline that analyzes restaurant reviews and classifies
them as Positive, Negative, or Neutral sentiment.

Tech Stack:
- NLTK — text preprocessing, tokenization, stopwords, lemmatization
- Scikit-learn — TF-IDF + Logistic Regression
- FastAPI — REST API backend
- Pandas, Matplotlib, Seaborn — data analysis and visualization

Author: Gourav Yadav
"""

import re
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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SAMPLE DATASET
SAMPLE_REVIEWS = [
    {"review": "The food was absolutely amazing! Best pasta I've ever had.", "sentiment": "positive"},
    {"review": "Great service and cozy atmosphere. Will definitely come back!", "sentiment": "positive"},
    {"review": "Delicious food, friendly staff, and reasonable prices.", "sentiment": "positive"},
    {"review": "Outstanding experience from start to finish. Highly recommended!", "sentiment": "positive"},
    {"review": "The pizza was perfectly crispy with fresh toppings. Loved it!", "sentiment": "positive"},
    {"review": "Excellent quality ingredients and the chef clearly knows his craft.", "sentiment": "positive"},
    {"review": "Wonderful dining experience. The desserts were heavenly!", "sentiment": "positive"},
    {"review": "Best restaurant in town! The steak was cooked to perfection.", "sentiment": "positive"},
    {"review": "Really enjoyed the ambiance and the food was top notch.", "sentiment": "positive"},
    {"review": "Fresh ingredients, quick service, and very affordable. Love it!", "sentiment": "positive"},
    {"review": "Terrible food. The pasta was overcooked and tasteless.", "sentiment": "negative"},
    {"review": "Very disappointed. Waited 45 minutes for cold food.", "sentiment": "negative"},
    {"review": "Rude staff and the portions were tiny for the price.", "sentiment": "negative"},
    {"review": "Never coming back. Found a hair in my soup.", "sentiment": "negative"},
    {"review": "Awful experience. The place was dirty and the food was stale.", "sentiment": "negative"},
    {"review": "Overpriced for what you get. Much better options nearby.", "sentiment": "negative"},
    {"review": "The worst dining experience of my life. Everything was wrong.", "sentiment": "negative"},
    {"review": "Food took forever and arrived cold. Completely unacceptable.", "sentiment": "negative"},
    {"review": "Disgusting food and horrible service. Avoid this place.", "sentiment": "negative"},
    {"review": "Extremely disappointing. The menu looked great but food was awful.", "sentiment": "negative"},
    {"review": "The food was okay, nothing special.", "sentiment": "neutral"},
    {"review": "Average experience. Food was decent but not memorable.", "sentiment": "neutral"},
    {"review": "It was fine. Not bad but not great either.", "sentiment": "neutral"},
    {"review": "Standard restaurant food. Service was acceptable.", "sentiment": "neutral"},
    {"review": "The meal was satisfactory. Prices are fair.", "sentiment": "neutral"},
    {"review": "Nothing to complain about but nothing to rave about either.", "sentiment": "neutral"},
    {"review": "Decent place for a quick meal. Nothing extraordinary.", "sentiment": "neutral"},
    {"review": "Food arrived on time and tasted normal. Average overall.", "sentiment": "neutral"},
]


# TEXT PREPROCESSING
class TextPreprocessor:
    """
    NLP text preprocessing pipeline using NLTK.

    Steps:
    1. Lowercase
    2. Remove special characters
    3. Tokenize
    4. Remove stopwords (keep sentiment words)
    5. Lemmatize
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep important sentiment words
        self.keep_words = {'not', 'no', 'never', 'very', 'too', 'most', 'more',
                           'but', 'however', 'although', 'worst', 'best'}
        self.stop_words -= self.keep_words

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def tokenize_and_filter(self, text: str) -> list:
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens
                  if t not in self.stop_words and len(t) > 2]
        return tokens

    def preprocess(self, text: str) -> str:
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return ' '.join(tokens)


# SENTIMENT CLASSIFIER


class SentimentClassifier:
    """
    Sentiment classifier using TF-IDF + Logistic Regression.
    Three classes: positive, negative, neutral.
    """

    def __init__(self):
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
                multi_class='multinomial',
                solver='lbfgs'
            ))
        ])
        self.is_trained = False

    def train(self, reviews: list) -> dict:
        df = pd.DataFrame(reviews)
        df['cleaned'] = df['review'].apply(self.preprocessor.preprocess)

        X = df['cleaned'].values
        y = df['sentiment'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Model trained. Accuracy: {accuracy:.2%}")
        return {"accuracy": round(accuracy, 4)}

    def predict(self, text: str) -> dict:
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

    def visualize(self, reviews: list):
        df = pd.DataFrame(reviews)
        df['predicted'] = df['review'].apply(lambda x: self.predict(x)['sentiment'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Restaurant Review Sentiment Analysis', fontsize=14, fontweight='bold')

        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        counts = df['predicted'].value_counts()
        bar_colors = [colors.get(s, '#3498db') for s in counts.index]

        axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor='white')
        axes[0].set_title('Sentiment Distribution')
        axes[0].set_xlabel('Sentiment')
        axes[0].set_ylabel('Count')

        axes[1].pie(counts.values, labels=counts.index, colors=bar_colors,
                    autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Sentiment Breakdown')

        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Chart saved: sentiment_distribution.png")



# FASTAPI APPLICATION

app = FastAPI(
    title="Restaurant Review Sentiment Analysis API",
    description="NLP-powered sentiment analysis using NLTK and Scikit-learn",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = SentimentClassifier()


@app.on_event("startup")
async def startup_event():
    metrics = classifier.train(SAMPLE_REVIEWS)
    logger.info(f"Model ready. Accuracy: {metrics['accuracy']:.2%}")


class ReviewRequest(BaseModel):
    review: str

    class Config:
        json_schema_extra = {
            "example": {"review": "The food was amazing and service was excellent!"}
        }

class BatchRequest(BaseModel):
    reviews: list[str]


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Restaurant Sentiment Analysis API",
        "docs": "Visit /docs for interactive API documentation"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_ready": classifier.is_trained
    }

@app.post("/analyze")
async def analyze(request: ReviewRequest):
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review cannot be empty.")
    return classifier.predict(request.review)

@app.post("/batch")
async def batch(request: BatchRequest):
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")

    results = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for review in request.reviews:
        if review.strip():
            result = classifier.predict(review)
            results.append(result)
            counts[result["sentiment"]] += 1

    return {
        "total": len(results),
        "summary": counts,
        "results": results
    }


# MAIN

if __name__ == "__main__":
    print("Training sentiment classifier...")
    clf = SentimentClassifier()
    metrics = clf.train(SAMPLE_REVIEWS)
    print(f"Accuracy: {metrics['accuracy']:.2%}")

    test_reviews = [
        "Absolutely loved the food! Best restaurant ever.",
        "Terrible experience. Food was cold and stale.",
        "The food was okay, nothing special really."
    ]

    print("\nSample Predictions:")
    for review in test_reviews:
        result = clf.predict(review)
        print(f"Review  : {review}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})")
        print()

    clf.visualize(SAMPLE_REVIEWS)
    print("Starting API server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

