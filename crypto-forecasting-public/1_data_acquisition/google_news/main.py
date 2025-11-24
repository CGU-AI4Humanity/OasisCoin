''' Script to fetch historic news data for a cryptocurrency from the Google News RSS feed. '''

import sys
sys.path.append('../../')

from functions import get_data

if __name__ == '__main__':

    print("=" * 60)
    print("Google News Data Acquisition - Bitcoin")
    print("=" * 60)
    print("\nThis will fetch Bitcoin news from 2017-10-01 to today")
    print("It may take a while as it processes each day...\n")
    print("Progress is being logged to google_news.log\n")

    try:
        print("Fetching Bitcoin news data...")
        btc_news_data = get_data('2017-10-01')
        btc_news_data.to_parquet(
            'btc_news_data.parquet.gzip', compression='gzip')
        print(f"✓ Saved {len(btc_news_data)} unique Bitcoin news articles")
        print(f"  (Duplicates removed during processing)")
    except Exception as e:
        print('❌ Error raised during bitcoin news data scraping: ', e)
        raise

    print("\n" + "=" * 60)
    print("✅ Data acquisition complete!")
    print("=" * 60)
