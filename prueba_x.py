
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

app = FastAPI()

# Función para limpiar el texto de los tweets
def clean_tweet(tweet: str) -> str:
    return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# Función para analizar el sentimiento usando TextBlob
def get_tweet_sentiment(tweet: str) -> str:
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Modelo de Pydantic para la entrada de datos
class TweetInput(BaseModel):
    tweets: List[str]

@app.post("/clean-tweets/")
def clean_tweets(data: TweetInput):
    cleaned_tweets = [clean_tweet(tweet) for tweet in data.tweets]
    return {"cleaned_tweets": cleaned_tweets}

@app.post("/analyze-sentiment/")
def analyze_sentiment(data: TweetInput):
    sentiments = [get_tweet_sentiment(tweet) for tweet in data.tweets]
    return {"sentiments": sentiments}

@app.post("/cluster-tweets/")
def cluster_tweets(data: TweetInput, n_clusters: int = 3):
    if not data.tweets:
        raise HTTPException(status_code=400, detail="No tweets provided.")
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data.tweets)
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    
    clusters = kmeans.labels_.tolist()
    return {"clusters": clusters}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
