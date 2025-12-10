# OasisCoin: Complete Step-by-Step Implementation Guide

**IST 332 Fall 2025 Final Project**  
**Group 4 - Claremont Graduate University**

---
## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Collection](#data-collection)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Sentiment Analysis](#sentiment-analysis)
6. [Topic Modeling](#topic-modeling)
7. [Feature Engineering](#feature-engineering)
8. [Model Training](#model-training)
9. [Model Evaluation](#model-evaluation)
10. [Trading Simulation](#trading-simulation)
11. [Results Analysis](#results-analysis)
12. [Troubleshooting](#troubleshooting)

---

## 1. Environment Setup

### Step 1.1: Prerequisites

Before starting, ensure you have:
- Python 3.12 installed (or Google Colab access)
- Approximately 4-5 GB of free disk space
- API keys for:
  - NewsAPI (https://newsapi.org/) - Free tier available
  - PRAW Reddit API (https://www.reddit.com/prefs/apps) - Free tier available
  - Binance API (optional, free for data collection)

### Step 1.2: Clone Repository

```bash
git clone https://github.com/CGU-AI4Humanity/OasisCoin.git
cd OasisCoin/oasis_coin_final
```

### Step 1.3: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv oasis_env
source oasis_env/bin/activate
```

**On Windows:**
```bash
python -m venv oasis_env
oasis_env\\Scripts\\activate
```

### Step 1.4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Manual installation:**
```bash
# Core data science
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install scikit-learn==1.4.2
pip install xgboost==2.0.3
pip install scipy==1.14.0

# NLP and Sentiment
pip install nltk==3.8.1
pip install textblob==0.17.1
pip install vaderSentiment==3.3.2

# APIs
pip install requests==2.31.0
pip install praw==7.7.1
pip install python-binance==1.0.19
pip install newsapi-python==0.2.7

# Visualization
pip install matplotlib==3.8.4
pip install seaborn==0.13.2
pip install plotly==5.22.0

# Jupyter
pip install jupyter==1.0.0
pip install ipykernel==6.29.4

# Utilities
pip install tqdm==4.66.4
pip install pytz==2024.1
```

### Step 1.5: Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
```

### Step 1.6: Verify Installation

```python
import pandas as pd
import numpy as np
import sklearn
import xgboost
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("âœ“ All packages installed successfully!")
print(f"  - pandas: {pd.__version__}")
print(f"  - numpy: {np.__version__}")
print(f"  - scikit-learn: {sklearn.__version__}")
print(f"  - xgboost: {xgboost.__version__}")
```

---

## 2. Data Collection

### Step 2.1: Set Up API Credentials

Create a `.env` file in your project directory:

```bash
# .env file (DO NOT commit to GitHub)
NEWSAPI_KEY=your_newsapi_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=OasisCoin/1.0 (by your_username)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
```

**Obtaining API Keys:**

1. **NewsAPI**: https://newsapi.org/ â†’ Sign up â†’ Copy API key â†’ Free: 100 requests/day
2. **Reddit PRAW**: https://www.reddit.com/prefs/apps â†’ Create app (script) â†’ Copy ID/Secret
3. **Binance**: https://www.binance.com/ â†’ Account â†’ API Management â†’ Create key

### Step 2.2: Load API Credentials

```python
import os
from dotenv import load_dotenv
import requests
import praw
from binance.client import Client
from newsapi import NewsApiClient

load_dotenv()

NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

print("âœ“ API credentials loaded successfully")
```

### Step 2.3: Collect Google News Articles

```python
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("COLLECTING GOOGLE NEWS ARTICLES")
print("=" * 80)

SEARCH_QUERY = "bitcoin"
DAYS_BACK = 180
START_DATE = (datetime.now() - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

all_articles = []
page = 1
max_pages = 100

print(f"Searching for: '{SEARCH_QUERY}'")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Collecting articles...")

try:
    while page <= max_pages:
        articles = newsapi.get_everything(
            q=SEARCH_QUERY,
            from_param=START_DATE,
            to=END_DATE,
            language='en',
            sort_by='publishedAt',
            page=page,
            page_size=100
        )
        
        if articles['status'] == 'ok':
            all_articles.extend(articles['articles'])
            print(f"  Page {page}: {len(articles['articles'])} articles")
            
            if len(articles['articles']) < 100:
                break
            page += 1
        else:
            print(f"Error: {articles['message']}")
            break
            
except Exception as e:
    print(f"Error: {e}")

news_df = pd.DataFrame([
    {
        'title': article['title'],
        'description': article['description'],
        'source': article['source']['name'],
        'url': article['url'],
        'publishedAt': article['publishedAt']
    }
    for article in all_articles
])

news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
news_df = news_df.drop_duplicates(subset=['title'])
news_df = news_df.sort_values('publishedAt')

print(f"\nâœ“ Collected {len(news_df)} unique articles")
news_df.to_csv('data/initial_datasets/google_news_bitcoin_2024_raw.csv', index=False)
```

### Step 2.4: Collect Reddit Comments

```python
print("=" * 80)
print("COLLECTING REDDIT COMMENTS")
print("=" * 80)

reddit_comments = []
subreddits = ['Bitcoin', 'CryptoCurrency']

for subreddit_name in subreddits:
    print(f"\nCollecting from r/{subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)
    
    post_count = 0
    comment_count = 0
    
    for submission in subreddit.new(limit=1000):
        post_count += 1
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            reddit_comments.append({
                'subreddit': subreddit_name,
                'author': str(comment.author) if comment.author else '[deleted]',
                'body': comment.body,
                'score': comment.score,
                'created_utc': datetime.fromtimestamp(comment.created_utc),
                'comment_id': comment.id
            })
            comment_count += 1
    
    print(f"  Posts: {post_count}, Comments: {comment_count}")

reddit_df = pd.DataFrame(reddit_comments)
reddit_df = reddit_df.drop_duplicates(subset=['comment_id'])

print(f"\nâœ“ Collected {len(reddit_df)} unique comments")
reddit_df.to_csv('data/initial_datasets/reddit_bitcoin_comments_raw.csv', index=False)
```

### Step 2.5: Collect Binance Price Data

```python
print("=" * 80)
print("COLLECTING BINANCE PRICE DATA")
print("=" * 80)

symbol = 'BTCUSDT'
interval = '1d'
start_date = datetime(2024, 3, 1)
end_date = datetime.now()

print(f"Collecting {symbol} candles from {start_date} to {end_date}...")

url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': symbol,
    'interval': interval,
    'startTime': int(start_date.timestamp() * 1000),
    'endTime': int(end_date.timestamp() * 1000),
    'limit': 1000
}

try:
    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 200:
        candles = response.json()
        price_df = pd.DataFrame([
            {
                'date': datetime.fromtimestamp(candle[0] / 1000),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[7])
            }
            for candle in candles
        ])
        print(f"âœ“ Downloaded {len(price_df)} candles")
except Exception as e:
    print(f"Error: {e}")

price_df.to_csv('data/initial_datasets/binance_btc_prices_2024_raw.csv', index=False)
print("âœ“ Price data saved")
```

---

## 3. Data Preprocessing

### Step 3.1: Load and Clean Data

```python
import pandas as pd
import numpy as np

print("=" * 80)
print("LOADING RAW DATA")
print("=" * 80)

news_df = pd.read_csv('data/initial_datasets/google_news_bitcoin_2024_raw.csv')
reddit_df = pd.read_csv('data/initial_datasets/reddit_bitcoin_comments_raw.csv')
price_df = pd.read_csv('data/initial_datasets/binance_btc_prices_2024_raw.csv')

news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
reddit_df['created_utc'] = pd.to_datetime(reddit_df['created_utc'])
price_df['date'] = pd.to_datetime(price_df['date'])

print(f"âœ“ Google News: {len(news_df)} records")
print(f"âœ“ Reddit: {len(reddit_df)} records")
print(f"âœ“ Price: {len(price_df)} records")
```

### Step 3.2: Text Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

print("\n" + "=" * 80)
print("TEXT PREPROCESSING")
print("=" * 80)

lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

crypto_stopwords = {
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'ethereum', 'eth',
    'blockchain', 'token', 'coin', 'price', 'market', 'trading', 'exchange',
    'buy', 'sell', 'said', 'would', 'could', 'one', 'two', 'year', 'time', 'day'
}
stopwords_to_use = english_stopwords | crypto_stopwords

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stopwords_to_use and len(token) > 2
    ]
    return tokens

print("Preprocessing Google News...")
news_df['title_tokens'] = news_df['title'].apply(preprocess_text)
news_df['title_clean'] = news_df['title_tokens'].apply(lambda x: ' '.join(x))
news_df = news_df[news_df['title_clean'].str.len() > 0]

print("Preprocessing Reddit...")
reddit_df['body_tokens'] = reddit_df['body'].apply(preprocess_text)
reddit_df['body_clean'] = reddit_df['body_tokens'].apply(lambda x: ' '.join(x))
reddit_df = reddit_df[reddit_df['body_clean'].str.len() > 0]

print(f"âœ“ News: {len(news_df)} records, Avg tokens: {news_df['title_tokens'].apply(len).mean():.1f}")
print(f"âœ“ Reddit: {len(reddit_df)} records, Avg tokens: {reddit_df['body_tokens'].apply(len).mean():.1f}")

news_df.to_parquet('data/processed/google_news_preprocessed.parquet.gzip', compression='gzip', index=False)
reddit_df.to_parquet('data/processed/reddit_comments_preprocessed.parquet.gzip', compression='gzip', index=False)
```

---

## 4. Exploratory Data Analysis

```python
print("=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

news_df['date'] = news_df['publishedAt'].dt.date
reddit_df['date'] = reddit_df['created_utc'].dt.date

news_daily = news_df.groupby('date').size()
reddit_daily = reddit_df.groupby('date').size()

print("\nTEMPORAL DISTRIBUTION:")
print(f"News: Min={news_daily.min()}, Max={news_daily.max()}, Mean={news_daily.mean():.1f}")
print(f"Reddit: Min={reddit_daily.min()}, Max={reddit_daily.max()}, Mean={reddit_daily.mean():.1f}")

print("\nTEXT STATISTICS:")
print(f"News sources: {news_df['source'].nunique()}")
print(f"Reddit authors: {reddit_df['author'].nunique()}")
print(f"Reddit avg score: {reddit_df['score'].mean():.2f}")

print("\nPRICE STATISTICS:")
print(f"Start: ${price_df['close'].iloc[0]:.2f}")
print(f"End: ${price_df['close'].iloc[-1]:.2f}")
print(f"Return: {((price_df['close'].iloc[-1] / price_df['close'].iloc[0]) - 1) * 100:.2f}%")

# Optional: Create visualizations
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
news_daily.plot(ax=axes[0], title='Google News Articles per Day', color='steelblue')
axes[0].set_ylabel('Count')
axes[0].grid(alpha=0.3)

reddit_daily.plot(ax=axes[1], title='Reddit Comments per Day', color='darkorange')
axes[1].set_ylabel('Count')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/temporal_distribution.png', dpi=150)
print("\nâœ“ Temporal plot saved")
```

---

## 5. Sentiment Analysis

### Step 5.1: Apply Sentiment

```python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("=" * 80)
print("SENTIMENT ANALYSIS")
print("=" * 80)

analyzer = SentimentIntensityAnalyzer()

print("\nApplying sentiment to Google News...")
news_df['textblob_polarity'] = news_df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
vader_scores = news_df['title'].apply(lambda x: analyzer.polarity_scores(str(x)))
news_df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])

print("Applying sentiment to Reddit...")
reddit_df['textblob_polarity'] = reddit_df['body'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
vader_scores_reddit = reddit_df['body'].apply(lambda x: analyzer.polarity_scores(str(x)))
reddit_df['vader_compound'] = vader_scores_reddit.apply(lambda x: x['compound'])

print("âœ“ Sentiment analysis complete")
```

### Step 5.2: Classify Sentiments

```python
def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

news_df['sentiment_class'] = news_df['vader_compound'].apply(classify_sentiment)
reddit_df['sentiment_class'] = reddit_df['vader_compound'].apply(classify_sentiment)

print("\nGoogle News Distribution:")
print(news_df['sentiment_class'].value_counts(normalize=True) * 100)

print("\nReddit Distribution:")
print(reddit_df['sentiment_class'].value_counts(normalize=True) * 100)

news_df.to_parquet('data/processed/google_news_with_sentiment.parquet.gzip', compression='gzip', index=False)
reddit_df.to_parquet('data/processed/reddit_with_sentiment.parquet.gzip', compression='gzip', index=False)
```

### Step 5.3: Aggregate by Day

```python
print("\nAggregating sentiment by day...")

news_daily_sentiment = news_df.groupby('date').agg({
    'vader_compound': ['mean', 'std'],
    'sentiment_class': lambda x: (x == 'positive').sum() / len(x) * 100
}).round(4)
news_daily_sentiment.columns = ['mean_sentiment', 'std_sentiment', 'pct_positive']

reddit_daily_sentiment = reddit_df.groupby('date').agg({
    'vader_compound': ['mean', 'std'],
    'sentiment_class': lambda x: (x == 'positive').sum() / len(x) * 100
}).round(4)
reddit_daily_sentiment.columns = ['mean_sentiment', 'std_sentiment', 'pct_positive']

price_df['date'] = pd.to_datetime(price_df['date']).dt.date

news_sentiment_price = news_daily_sentiment.reset_index().merge(price_df, on='date', how='inner')
reddit_sentiment_price = reddit_daily_sentiment.reset_index().merge(price_df, on='date', how='inner')

print(f"âœ“ Merged: News {len(news_sentiment_price)} days, Reddit {len(reddit_sentiment_price)} days")

news_sentiment_price.to_parquet('data/processed/news_price_merged.parquet.gzip', compression='gzip', index=False)
reddit_sentiment_price.to_parquet('data/processed/reddit_price_merged.parquet.gzip', compression='gzip', index=False)
```

---

## 6. Topic Modeling

### Step 6.1: Vectorize Text

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

print("=" * 80)
print("TOPIC MODELING")
print("=" * 80)

print("\nVectorizing Google News...")
news_tfidf = TfidfVectorizer(max_features=100, min_df=5, max_df=0.8).fit_transform(news_df['title_clean'])
news_count = CountVectorizer(max_features=100, min_df=5, max_df=0.8).fit_transform(news_df['title_clean'])

print("Vectorizing Reddit...")
reddit_tfidf = TfidfVectorizer(max_features=200, min_df=10, max_df=0.8).fit_transform(reddit_df['body_clean'])
reddit_count = CountVectorizer(max_features=200, min_df=10, max_df=0.8).fit_transform(reddit_df['body_clean'])

print(f"âœ“ News TF-IDF: {news_tfidf.shape}, Reddit TF-IDF: {reddit_tfidf.shape}")
```

### Step 6.2: Train LDA & NMF

```python
print("\nTraining LDA models...")
lda_news = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=20)
lda_news_topics = lda_news.fit_transform(news_count)

lda_reddit = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=20)
lda_reddit_topics = lda_reddit.fit_transform(reddit_count)

print("Training NMF models...")
nmf_news = NMF(n_components=8, random_state=42, max_iter=500, init='nndsvd')
nmf_news_topics = nmf_news.fit_transform(news_tfidf)

nmf_reddit = NMF(n_components=8, random_state=42, max_iter=500, init='nndsvd')
nmf_reddit_topics = nmf_reddit.fit_transform(reddit_tfidf)

print("âœ“ Topic models trained")

# Display topics
def display_topics(model, vectorizer, n_words=10):
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

print("\nGoogle News LDA Topics:")
display_topics(lda_news, news_count_vectorizer)
```

---

## 7. Feature Engineering

### Step 7.1: Create Features

```python
print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

news_sentiment_price = pd.read_parquet('data/processed/news_price_merged.parquet.gzip')
reddit_sentiment_price = pd.read_parquet('data/processed/reddit_price_merged.parquet.gzip')

def engineer_price_features(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['momentum_3d'] = df['close'].pct_change(3)
    df['momentum_7d'] = df['close'].pct_change(7)
    df['volatility_5d'] = df['returns'].rolling(window=5).std()
    df['volatility_10d'] = df['returns'].rolling(window=10).std()
    df['next_day_return'] = df['returns'].shift(-1)
    df['price_up_tomorrow'] = (df['next_day_return'] > 0).astype(int)
    return df

def engineer_temporal_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['date'].dt.day <= 3).astype(int)
    df['is_month_end'] = (df['date'].dt.day >= 28).astype(int)
    return df

news_sentiment_price = engineer_price_features(news_sentiment_price)
news_sentiment_price = engineer_temporal_features(news_sentiment_price)

reddit_sentiment_price = engineer_price_features(reddit_sentiment_price)
reddit_sentiment_price = engineer_temporal_features(reddit_sentiment_price)

print("âœ“ Features created")

news_sentiment_price.to_parquet('data/processed/news_price_engineered.parquet.gzip', compression='gzip', index=False)
reddit_sentiment_price.to_parquet('data/processed/reddit_price_engineered.parquet.gzip', compression='gzip', index=False)
```

---

## 8. Model Training

### Step 8.1: Prepare Data

```python
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("MODEL TRAINING")
print("=" * 80)

news_data = pd.read_parquet('data/processed/news_price_engineered.parquet.gzip')
reddit_data = pd.read_parquet('data/processed/reddit_price_engineered.parquet.gzip')

feature_columns = [
    'mean_sentiment', 'std_sentiment', 'pct_positive',
    'momentum_3d', 'momentum_7d', 'volatility_5d', 'volatility_10d',
    'day_of_week', 'month', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end'
]

X_news = news_data[feature_columns].fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)
y_news = news_data['price_up_tomorrow'].dropna()
X_news = X_news[:len(y_news)]

X_reddit = reddit_data[feature_columns].fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)
y_reddit = reddit_data['price_up_tomorrow'].dropna()
X_reddit = X_reddit[:len(y_reddit)]

test_size = int(len(X_news) * 0.2)

X_news_train, X_news_test = X_news[:-test_size], X_news[-test_size:]
y_news_train, y_news_test = y_news[:-test_size], y_news[-test_size:]

X_reddit_train, X_reddit_test = X_reddit[:-test_size], X_reddit[-test_size:]
y_reddit_train, y_reddit_test = y_reddit[:-test_size], y_reddit[-test_size:]

scaler_news = StandardScaler()
X_news_train_scaled = scaler_news.fit_transform(X_news_train)
X_news_test_scaled = scaler_news.transform(X_news_test)

scaler_reddit = StandardScaler()
X_reddit_train_scaled = scaler_reddit.fit_transform(X_reddit_train)
X_reddit_test_scaled = scaler_reddit.transform(X_reddit_test)

print(f"âœ“ Data prepared: News {len(X_news_train)}/{len(X_news_test)}, Reddit {len(X_reddit_train)}/{len(X_reddit_test)}")
```

### Step 8.2: Train Models

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("\nTraining models...")

lr = GridSearchCV(
    LogisticRegression(solver='lbfgs', random_state=42),
    {'C': [0.01, 0.1, 1.0], 'max_iter': [1000]},
    cv=5, scoring='f1'
)
lr.fit(X_news_train_scaled, y_news_train)

rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
    cv=5, scoring='f1'
)
rf.fit(X_news_train, y_news_train)

xgb_model = GridSearchCV(
    xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
    cv=5, scoring='f1'
)
xgb_model.fit(X_news_train, y_news_train)

print("âœ“ Google News models trained")

lr_reddit = GridSearchCV(LogisticRegression(solver='lbfgs', random_state=42),
                         {'C': [0.01, 0.1, 1.0], 'max_iter': [1000]}, cv=5, scoring='f1')
lr_reddit.fit(X_reddit_train_scaled, y_reddit_train)

rf_reddit = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                         {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
                         cv=5, scoring='f1')
rf_reddit.fit(X_reddit_train, y_reddit_train)

xgb_reddit = GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                          {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
                          cv=5, scoring='f1')
xgb_reddit.fit(X_reddit_train, y_reddit_train)

print("âœ“ Reddit models trained")

# Save models
import pickle
pickle.dump(xgb_model, open('models/xgboost_news_model.pkl', 'wb'))
pickle.dump(xgb_reddit, open('models/xgboost_reddit_model.pkl', 'wb'))
```

---

## 9. Model Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    if scaler:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}, AUC: {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

print("\nGOOGLE NEWS:")
news_lr = evaluate_model(lr, X_news_test, y_news_test, "Logistic Regression", scaler_news)
news_rf = evaluate_model(rf, X_news_test, y_news_test, "Random Forest")
news_xgb = evaluate_model(xgb_model, X_news_test, y_news_test, "XGBoost")

print("\nREDDIT:")
reddit_lr = evaluate_model(lr_reddit, X_reddit_test, y_reddit_test, "Logistic Regression", scaler_reddit)
reddit_rf = evaluate_model(rf_reddit, X_reddit_test, y_reddit_test, "Random Forest")
reddit_xgb = evaluate_model(xgb_reddit, X_reddit_test, y_reddit_test, "XGBoost")

# Create comparison table
results_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'News Accuracy': [news_lr['accuracy'], news_rf['accuracy'], news_xgb['accuracy']],
    'News F1': [news_lr['f1'], news_rf['f1'], news_xgb['f1']],
    'Reddit Accuracy': [reddit_lr['accuracy'], reddit_rf['accuracy'], reddit_xgb['accuracy']],
    'Reddit F1': [reddit_lr['f1'], reddit_rf['f1'], reddit_xgb['f1']]
})

print("\n\nMODEL COMPARISON:")
print(results_comparison.to_string(index=False))
```

---

## 10. Trading Simulation

```python
print("=" * 80)
print("TRADING SIMULATION")
print("=" * 80)

def run_trading_strategy(model, X_data, y_data, prices, scaler=None, 
                        threshold_up=0.60, threshold_down=0.40, initial_capital=10000):
    if scaler:
        X_scaled = scaler.transform(X_data)
        predictions = model.predict_proba(X_scaled)[:, 1]
    else:
        predictions = model.predict_proba(X_data)[:, 1]
    
    portfolio_value = initial_capital
    position = 'cash'
    entry_price = None
    portfolio_values = [initial_capital]
    trades = []
    
    for i in range(len(predictions)):
        signal = predictions[i]
        price = prices[i]
        
        if signal > threshold_up and position == 'cash':
            entry_price = price
            position = 'long'
            trades.append({'entry_date': i, 'entry_price': price})
        
        elif signal < threshold_down and position == 'long':
            exit_price = price
            profit = (exit_price - entry_price) / entry_price
            portfolio_value *= (1 + profit)
            trades[-1]['exit_date'] = i
            trades[-1]['exit_price'] = exit_price
            trades[-1]['return'] = profit
            position = 'cash'
        
        portfolio_values.append(portfolio_value)
    
    if position == 'long':
        final_price = prices[-1]
        profit = (final_price - entry_price) / entry_price
        portfolio_value *= (1 + profit)
        trades[-1]['exit_date'] = len(prices) - 1
        trades[-1]['exit_price'] = final_price
        trades[-1]['return'] = profit
    
    return portfolio_value, portfolio_values, trades

print("\nGoogle News Trading...")
final_value_news, portfolio_values_news, trades_news = run_trading_strategy(
    xgb_model, X_news_test, y_news_test,
    news_data['close'].iloc[-len(X_news_test):].values,
    scaler=scaler_news
)

print(f"âœ“ Final: ${final_value_news:.2f}, Return: {((final_value_news - 10000) / 10000) * 100:.2f}%")
print(f"  Total trades: {len(trades_news)}")

print("\nReddit Trading...")
final_value_reddit, portfolio_values_reddit, trades_reddit = run_trading_strategy(
    xgb_reddit, X_reddit_test, y_reddit_test,
    reddit_data['close'].iloc[-len(X_reddit_test):].values,
    scaler=scaler_reddit
)

print(f"âœ“ Final: ${final_value_reddit:.2f}, Return: {((final_value_reddit - 10000) / 10000) * 100:.2f}%")
print(f"  Total trades: {len(trades_reddit)}")
```

---

## 11. Results Analysis

```python
print("=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

def calculate_metrics(portfolio_values, trades, initial_capital=10000):
    metrics = {}
    metrics['total_return'] = ((portfolio_values[-1] - initial_capital) / initial_capital) * 100
    
    portfolio_array = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdown = (portfolio_array - running_max) / running_max
    metrics['max_drawdown'] = np.min(drawdown) * 100
    
    winning_trades = sum(1 for trade in trades if trade.get('return', 0) > 0)
    metrics['win_rate'] = (winning_trades / len(trades)) * 100 if trades else 0
    
    returns = np.diff(portfolio_array) / portfolio_array[:-1]
    metrics['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    return metrics

news_metrics = calculate_metrics(portfolio_values_news, trades_news)
reddit_metrics = calculate_metrics(portfolio_values_reddit, trades_reddit)

news_bh = (news_data['close'].iloc[-1] / news_data['close'].iloc[0] - 1) * 100
reddit_bh = (reddit_data['close'].iloc[-1] / reddit_data['close'].iloc[0] - 1) * 100

print("\nGOOGLE NEWS:")
print(f"  Strategy: {news_metrics['total_return']:.2f}% | Buy-Hold: {news_bh:.2f}%")
print(f"  Underperformance: {news_metrics['total_return'] - news_bh:.2f}%")
print(f"  Max Drawdown: {news_metrics['max_drawdown']:.2f}% | Sharpe: {news_metrics['sharpe_ratio']:.2f}")
print(f"  Win Rate: {news_metrics['win_rate']:.1f}%")

print("\nREDDIT:")
print(f"  Strategy: {reddit_metrics['total_return']:.2f}% | Buy-Hold: {reddit_bh:.2f}%")
print(f"  Underperformance: {reddit_metrics['total_return'] - reddit_bh:.2f}%")
print(f"  Max Drawdown: {reddit_metrics['max_drawdown']:.2f}% | Sharpe: {reddit_metrics['sharpe_ratio']:.2f}")
print(f"  Win Rate: {reddit_metrics['win_rate']:.1f}%")

# Visualize results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].plot(portfolio_values_news, linewidth=2, color='steelblue')
axes[0, 0].axhline(y=10000, color='gray', linestyle='--')
axes[0, 0].set_title('Google News: Portfolio Value Over Time', fontweight='bold')
axes[0, 0].set_ylabel('Portfolio Value')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(portfolio_values_reddit, linewidth=2, color='darkorange')
axes[0, 1].axhline(y=10000, color='gray', linestyle='--')
axes[0, 1].set_title('Reddit: Portfolio Value Over Time', fontweight='bold')
axes[0, 1].set_ylabel('Portfolio Value')
axes[0, 1].grid(alpha=0.3)

metrics_names = ['Return (%)', 'Max Drawdown (%)', 'Win Rate (%)', 'Sharpe Ratio']
news_metrics_values = [
    news_metrics['total_return'],
    abs(news_metrics['max_drawdown']),
    news_metrics['win_rate'],
    news_metrics['sharpe_ratio'] * 10
]

axes[1, 0].bar(metrics_names, news_metrics_values, color=['green' if v > 0 else 'red' for v in news_metrics_values])
axes[1, 0].set_title('Google News: Trading Metrics', fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

reddit_metrics_values = [
    reddit_metrics['total_return'],
    abs(reddit_metrics['max_drawdown']),
    reddit_metrics['win_rate'],
    reddit_metrics['sharpe_ratio'] * 10
]

axes[1, 1].bar(metrics_names, reddit_metrics_values, color=['green' if v > 0 else 'red' for v in reddit_metrics_values])
axes[1, 1].set_title('Reddit: Trading Metrics', fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/trading_simulation_results.png', dpi=150)
print("\nâœ“ Results visualization saved")
```

---

## 12. Troubleshooting

### Common Issues & Solutions

**Issue 1: API Rate Limits**
```python
import time
time.sleep(1)  # Add delays between API calls
```

**Issue 2: Memory Issues with Large Datasets**
```python
chunk_size = 10000
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    # Process chunk
```

**Issue 3: NaN Values in Features**
```python
df = df.fillna(0)  # Fill with zero
# OR
df = df.dropna()  # Drop NaN rows
```

**Issue 4: Model Not Converging**
```python
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model_xgb = xgb.XGBClassifier(learning_rate=0.01, n_estimators=500)
```

**Issue 5: Data Leakage in Time Series**
```python
# DO NOT shuffle time series data
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]
```

**Issue 6: Connection Timeouts**
```python
import requests
response = requests.get(url, params=params, timeout=30)
```

**Issue 7: Infinite Values in Data**
```python
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
```

---

## Summary

This guide covers the entire OasisCoin pipeline:

âœ… Environment setup and dependency installation  
âœ… Data collection (49,758 total records)  
âœ… Text preprocessing and tokenization  
âœ… Exploratory data analysis  
âœ… Sentiment analysis (TextBlob + VADER)  
âœ… Topic modeling (LDA + NMF)  
âœ… Feature engineering (77+ features)  
âœ… Model training (3 models Ã— 2 sources)  
âœ… Model evaluation and comparison  
âœ… Trading simulation and backtesting  
âœ… Results analysis and visualization  
âœ… Troubleshooting common issues  

**Expected Runtime:** 3-4 hours on Google Colab GPU

**Key Outputs:**
- `data/processed/` - Preprocessed and engineered data
- `models/` - Trained model artifacts
- `results/` - Visualizations and metrics

**Key Findings:**
- Classification models: 57-60% accuracy
- Trading returns: News 9.09%, Reddit 0.82%
- Buy-and-hold returns: News 49.99%, Reddit 34.35%
- Sentiment diversity predicts volatility (r=0.19-0.21, p<0.01)

---

**Good luck with your presentation! ðŸš€**
