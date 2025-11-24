''' Functions for Google News data fetching. '''

import sys
sys.path.append('../../')

import re
import string
import random
import requests
import pandas as pd
import logging
import os
from pathlib import Path

from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from functools import lru_cache

# Set up logging
log_file = Path(__file__).parent / 'google_news.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)


class GoogleNewsRSS:
    ''' Scraper for Google News RSS. '''

    def __init__(self, rss_url):

        # Fetch url
        self.response = requests.get(rss_url)
        
        # Log HTTP response status
        if self.response.status_code != 200:
            logger.warning(f'HTTP {self.response.status_code} for URL: {rss_url}')

        # Parse response (use 'xml' parser for RSS feeds)
        try:
            self.soup = BeautifulSoup(self.response.text, 'xml')
        except Exception as e:
            logger.error(f'Could not parse the xml: {rss_url}')
            logger.error(f'Error: {e}')

        # Extract individual elements of google news and some metadata
        self.articles = self.soup.findAll('item')
        self.size = len(self.articles)

        self.articles_dicts = []
        for a in self.articles:
            try:
                title_elem = a.find('title')
                link_elem = a.find('link')
                desc_elem = a.find('description')
                
                # Try multiple date tag names (pubDate with capital D is the correct one)
                pubdate_elem = None
                for date_tag in ['pubDate', 'pubdate', 'published', 'dc:date', 'date']:
                    pubdate_elem = a.find(date_tag)
                    if pubdate_elem and pubdate_elem.text:
                        break
                
                # Handle link - try different methods
                link = ''
                if link_elem:
                    if link_elem.next_sibling:
                        link = str(link_elem.next_sibling).replace('\n', '').replace('\t', '').strip()
                    elif link_elem.text:
                        link = link_elem.text.strip()
                
                # Get date - try text or string
                pubdate_text = ''
                if pubdate_elem:
                    pubdate_text = pubdate_elem.text if pubdate_elem.text else str(pubdate_elem).strip()
                
                article_dict = {
                    'title': title_elem.text if title_elem else '',
                    'link': link,
                    'description': desc_elem.text if desc_elem else '',
                    'pubdate': pubdate_text
                }
                self.articles_dicts.append(article_dict)
            except Exception as e:
                # Skip articles that can't be parsed
                continue

        self.urls = [d['link'] for d in self.articles_dicts if 'link' in d]
        self.titles = [d['title'] for d in self.articles_dicts if 'title' in d]
        self.descriptions = [d['description']
                             for d in self.articles_dicts if 'description' in d]
        self.publication_times = [d['pubdate']
                                  for d in self.articles_dicts if 'pubdate' in d]


@lru_cache
def convert_time(time: str):
    ''' Convert Google News date string to datetime. '''
    if not time or time.strip() == '':
        return None
    
    # Try multiple date formats that Google News RSS might use
    date_formats = [
        '%d %b %Y %H:%M:%S',  # Original format: "01 Oct 2017 07:00:00"
        '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822: "Mon, 01 Oct 2017 07:00:00 GMT"
        '%a, %d %b %Y %H:%M:%S %z',  # With timezone offset
        '%Y-%m-%d %H:%M:%S',  # ISO-like
        '%Y-%m-%dT%H:%M:%S',  # ISO format
        '%Y-%m-%dT%H:%M:%SZ',  # ISO with Z
        '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO with microseconds
    ]
    
    # Clean the date string
    time_clean = time.strip()
    
    # Try original method first (for backward compatibility)
    try:
        _format = '%d %b %Y %H:%M:%S'
        cleaned = re.sub(r'^.*?,', ',', time_clean)[2:][:-4]
        return datetime.strptime(cleaned, _format)
    except:
        pass
    
    # Try other formats
    for fmt in date_formats:
        try:
            return datetime.strptime(time_clean, fmt)
        except:
            continue
    
    # If all formats fail, return None
    return None


def get_data(start_date: str = '2017-10-01'):
    ''' Fetch Bitcoin news data from Google News RSS feed.
    
    Starts from current date and goes backward until start_date is reached.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format (default: '2017-10-01')
    '''

    link = string.Template(
        'https://news.google.com/rss/search?q=CoinDesk+OR+Cointelegraph+OR+Decrypt,+Bitcoin+OR+BTC+after:$early_date+before:$late_date&ceid=US:en&hl=en-US&gl=US')

    currency = 'Bitcoin'
    coin = 'BTC'

    # Parse start date
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    start_date_obj = start_datetime.date()

    logger.info(f"=" * 60)
    logger.info(f"Starting Bitcoin news data acquisition")
    logger.info(f"Date range: {start_date} to today")
    logger.info(f"=" * 60)
    
    all_data = pd.DataFrame()
    
    # Start from today and go backward
    current_date = datetime.now().date()
    days_processed = 0
    
    logger.info(f"Starting from {current_date} and going backward to {start_date}...")
    logger.info(f"")

    while current_date >= start_date_obj:
        # Calculate date range for this day
        date = datetime.combine(current_date, datetime.min.time())
        next_date = date + timedelta(days=1)

        # Request data from Google News
        URL = link.substitute(currency=currency,
                              symbol=coin,
                              early_date=date.strftime('%Y-%m-%d'),
                              late_date=next_date.strftime('%Y-%m-%d'))
        
        try:
            request = GoogleNewsRSS(URL)

            response = [request.publication_times, request.titles, request.urls]
            c_data = pd.DataFrame(response).T
            c_data.columns = ['time', 'title', 'url']
            
            # Convert times and filter out None values
            c_data['datetime'] = [convert_time(i) for i in c_data.time]
            # Remove rows where datetime conversion failed
            c_data = c_data[c_data['datetime'].notna()]
            
            if len(c_data) > 0:
                c_data['timestamp'] = [datetime.timestamp(i) for i in c_data.datetime]
                c_data = c_data.drop(columns='time').set_index('timestamp')
                all_data = pd.concat([all_data, c_data])
                logger.info(f"Date: {current_date} | Articles fetched: {len(c_data)} | Total articles: {len(all_data)}")
            else:
                logger.info(f"Date: {current_date} | Articles fetched: 0 | Total articles: {len(all_data)}")
                
            # Summary update every 100 days
            if days_processed % 100 == 0 and days_processed > 0:
                logger.info(f"--- Summary: {days_processed} days processed | Total articles: {len(all_data)} | Current date: {current_date} ---")
        except Exception as e:
            logger.error(f"Date: {current_date} | ERROR: {e}")
        
        # Move to previous day
        current_date = current_date - timedelta(days=1)
        days_processed += 1
        
        # Small delay to avoid rate limiting
        if days_processed % 10 == 0:
            import time
            time.sleep(0.5)

    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"Data acquisition complete for Bitcoin")
    logger.info(f"Total articles fetched: {len(all_data)}")
    logger.info(f"Days processed: {days_processed}")
    logger.info(f"Date range: {start_date} to {datetime.now().date()}")
    logger.info(f"=" * 60)
    
    return all_data.sort_index().drop_duplicates()
