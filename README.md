# Sentiment Analysis Tool

A powerful sentiment analysis tool built with Python that analyzes text sentiment using state-of-the-art open-source models. This tool provides comprehensive text analysis capabilities including sentiment classification, emotion detection, and sentiment intensity analysis.

## Features

- **Sentiment Analysis**: Classifies text into positive or negative sentiment using the DistilBERT model
- **Emotion Detection**: Identifies specific emotions (joy, anger, sadness, etc.) using the DistilRoBERTa model
- **Sentiment Intensity**: Measures the strength of sentiment using TextBlob
- **Interactive Visualizations**: Generates three types of interactive HTML visualizations:
  - Sentiment distribution pie chart
  - Emotion distribution bar chart
  - Sentiment intensity histogram
- **Batch Processing**: Analyze multiple texts at once
- **GPU Acceleration**: Automatically utilizes GPU if available for faster processing

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sentimental-ai
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python sentiment_analyzer.py
```

2. The script will:
   - Analyze sample texts
   - Generate interactive visualizations
   - Save the visualizations as HTML files:
     - `sentiment_distribution.html`
     - `emotion_distribution.html`
     - `intensity_distribution.html`

3. To analyze your own texts, modify the `sample_texts` list in the `main()` function of `sentiment_analyzer.py`.

## Code Example

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Analyze a single text
text = "I absolutely love this product! It's amazing."
sentiment = analyzer.analyze_sentiment(text)
emotion = analyzer.analyze_emotion(text)
intensity = analyzer.get_sentiment_intensity(text)

# Analyze multiple texts
texts = ["Great product!", "Terrible service", "It's okay"]
results_df = analyzer.analyze_batch(texts)

# Generate visualizations
fig1, fig2, fig3 = analyzer.visualize_sentiment_distribution(results_df)
```

## Dependencies

- **transformers**: For using pre-trained models (DistilBERT and DistilRoBERTa)
- **torch**: Deep learning framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **nltk**: Natural Language Processing tools
- **textblob**: Text processing and sentiment analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities

## Models Used

- **Sentiment Analysis**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Emotion Analysis**: `j-hartmann/emotion-english-distilroberta-base`

## Output

The tool generates three interactive HTML visualizations:
1. A pie chart showing the distribution of positive and negative sentiments
2. A bar chart displaying the distribution of different emotions
3. A histogram showing the distribution of sentiment intensity scores

## License

MIT License 