import pandas as pd
from textblob import TextBlob

# Load dataset
df = pd.read_csv("data/social_media_data.csv")

def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df.to_csv("data/sentiment_results.csv", index=False)
print(df.head())




