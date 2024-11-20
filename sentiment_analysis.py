from transformers import pipeline

# Sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")
text = "The character's determination inspired everyone around her."
result = sentiment_analyzer(text)
print(result)  # [{'label': 'POSITIVE', 'score': 0.98}]
