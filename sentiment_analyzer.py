import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class SentimentAnalyzer:
    def __init__(self):
        # Initialize the sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize emotion analysis pipeline
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.stop_words = set(stopwords.words('english'))

    def analyze_sentiment(self, text):
        """Analyze the sentiment of a given text."""
        result = self.sentiment_pipeline(text)[0]
        return {
            'sentiment': result['label'],
            'score': result['score']
        }

    def analyze_emotion(self, text):
        """Analyze the emotion in the text."""
        result = self.emotion_pipeline(text)[0]
        return {
            'emotion': result['label'],
            'score': result['score']
        }

    def get_sentiment_intensity(self, text):
        """Calculate sentiment intensity using TextBlob."""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def analyze_batch(self, texts):
        """Analyze a batch of texts and return comprehensive results."""
        results = []
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            emotion = self.analyze_emotion(text)
            intensity = self.get_sentiment_intensity(text)
            
            results.append({
                'text': text,
                'sentiment': sentiment['sentiment'],
                'sentiment_score': sentiment['score'],
                'emotion': emotion['emotion'],
                'emotion_score': emotion['score'],
                'intensity': intensity
            })
        return pd.DataFrame(results)

    def visualize_sentiment_distribution(self, df):
        """Create interactive visualizations of sentiment distribution."""
        # Sentiment distribution pie chart
        sentiment_counts = df['sentiment'].value_counts()
        fig1 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Sentiment Distribution'
        )

        # Emotion distribution bar chart
        emotion_counts = df['emotion'].value_counts()
        fig2 = px.bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            title='Emotion Distribution',
            labels={'x': 'Emotion', 'y': 'Count'}
        )

        # Sentiment intensity histogram
        fig3 = px.histogram(
            df,
            x='intensity',
            title='Sentiment Intensity Distribution',
            nbins=20
        )

        return fig1, fig2, fig3

def main():
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Sample texts for analysis
    sample_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This movie was terrible, I want my money back.",
        "The service was okay, nothing special.",
        "I'm so excited about this new feature!",
        "This is the worst experience ever."
    ]
    
    # Analyze the texts
    results_df = analyzer.analyze_batch(sample_texts)
    
    # Print results
    print("\nAnalysis Results:")
    print(results_df)
    
    # Create visualizations
    fig1, fig2, fig3 = analyzer.visualize_sentiment_distribution(results_df)
    
    # Save visualizations
    fig1.write_html("sentiment_distribution.html")
    fig2.write_html("emotion_distribution.html")
    fig3.write_html("intensity_distribution.html")

if __name__ == "__main__":
    main() 