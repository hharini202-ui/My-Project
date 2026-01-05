import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load results
df = pd.read_csv("data/sentiment_results.csv")

# Sentiment distribution
sns.countplot(x="Sentiment", data=df, palette="coolwarm")
plt.title("Overall Sentiment Distribution")
plt.show()

# Sentiment by Platform
sns.countplot(x="Platform", hue="Sentiment", data=df, palette="Set2")
plt.title("Sentiment by Platform")
plt.show()

# Sentiment by Country
sns.countplot(x="Country", hue="Sentiment", data=df, palette="Set3")
plt.title("Sentiment by Country")
plt.xticks(rotation=45)
plt.show()

# Word Cloud of all posts
text = " ".join(df["Text"].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Trend over time (Sentiment counts per day)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
daily_sentiment = df.groupby([df["Timestamp"].dt.date, "Sentiment"]).size().unstack(fill_value=0)
daily_sentiment.plot(kind="line", marker="o")
plt.title("Daily Sentiment Trend")
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

