"""
News sentiment analysis for trading signals.
Integrates with various news APIs to analyze market sentiment.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import json
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

@dataclass
class NewsConfig:
    alpha_vantage_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    enable_sentiment: bool = True
    sentiment_window_days: int = 7
    cache_duration_minutes: int = 60

class NewsSentimentAnalyzer:
    """Analyzes news sentiment for trading signals."""

    def __init__(self, config: NewsConfig = None):
        if config is None:
            config = NewsConfig(
                alpha_vantage_key=os.getenv('ALPHA_VANTAGE_KEY'),
                newsapi_key=os.getenv('NEWSAPI_KEY')
            )
        self.config = config
        self.sentiment_cache = {}
        self.vader = SentimentIntensityAnalyzer()

    def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict[str, float]:
        """Get sentiment scores for a symbol over the last N days."""
        cache_key = f"{symbol}_{days}"

        # Check cache
        if cache_key in self.sentiment_cache:
            cached_time, cached_data = self.sentiment_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.config.cache_duration_minutes * 60:
                return cached_data

        sentiment_scores = self._fetch_news_sentiment(symbol, days)

        # Cache results
        self.sentiment_cache[cache_key] = (datetime.now(), sentiment_scores)

        return sentiment_scores

    def _fetch_news_sentiment(self, symbol: str, days: int) -> Dict[str, float]:
        """Fetch and analyze news for sentiment."""
        all_articles = []

        # Try Alpha Vantage first
        if self.config.alpha_vantage_key:
            articles = self._get_alpha_vantage_news(symbol, days)
            all_articles.extend(articles)

        # Try NewsAPI as backup
        if self.config.newsapi_key and len(all_articles) < 5:
            articles = self._get_newsapi_news(symbol, days)
            all_articles.extend(articles)

        if not all_articles:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'article_count': 0}

        # Analyze sentiment
        sentiments = []
        for article in all_articles:
            title_sentiment = self._analyze_sentiment(article.get('title', ''))
            desc_sentiment = self._analyze_sentiment(article.get('description', ''))
            # Weight title more heavily
            combined = {
                'compound': (title_sentiment['compound'] * 0.7 + desc_sentiment['compound'] * 0.3),
                'positive': (title_sentiment['positive'] * 0.7 + desc_sentiment['positive'] * 0.3),
                'negative': (title_sentiment['negative'] * 0.7 + desc_sentiment['negative'] * 0.3),
                'neutral': (title_sentiment['neutral'] * 0.7 + desc_sentiment['neutral'] * 0.3)
            }
            sentiments.append(combined)

        # Aggregate sentiments
        if sentiments:
            avg_sentiment = {
                'compound': np.mean([s['compound'] for s in sentiments]),
                'positive': np.mean([s['positive'] for s in sentiments]),
                'negative': np.mean([s['negative'] for s in sentiments]),
                'neutral': np.mean([s['neutral'] for s in sentiments]),
                'article_count': len(sentiments)
            }
        else:
            avg_sentiment = {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'article_count': 0}

        return avg_sentiment

    def _get_alpha_vantage_news(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from Alpha Vantage."""
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=financial_markets&apikey={self.config.alpha_vantage_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            articles = []
            if 'feed' in data:
                cutoff_date = datetime.now() - timedelta(days=days)
                for article in data['feed'][:20]:  # Limit to 20 articles
                    pub_date = datetime.fromisoformat(article['time_published'][:-1])
                    if pub_date >= cutoff_date:
                        # Check if article mentions the symbol
                        text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
                        if symbol.lower() in text or any(ticker in text for ticker in [symbol, f"${symbol}"]):
                            articles.append({
                                'title': article.get('title', ''),
                                'description': article.get('summary', ''),
                                'published_at': pub_date
                            })
            return articles
        except Exception as e:
            print(f"Alpha Vantage news error: {e}")
            return []

    def _get_newsapi_news(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from NewsAPI."""
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"https://newsapi.org/v2/everything?q={symbol}&from={from_date}&sortBy=publishedAt&apiKey={self.config.newsapi_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            articles = []
            if data.get('status') == 'ok':
                for article in data.get('articles', [])[:10]:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'published_at': article.get('publishedAt', '')
                    })
            return articles
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using VADER."""
        if not text or not isinstance(text, str):
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

        # Use VADER for more accurate financial sentiment
        scores = self.vader.polarity_scores(text)

        return scores

def get_market_sentiment_index(symbols: List[str], days: int = 7) -> pd.Series:
    """Get market sentiment index for multiple symbols."""
    analyzer = NewsSentimentAnalyzer()

    sentiments = {}
    for symbol in symbols:
        sentiment = analyzer.get_news_sentiment(symbol, days)
        sentiments[symbol] = sentiment['compound']

    return pd.Series(sentiments)