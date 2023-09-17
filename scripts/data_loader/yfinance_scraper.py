from scripts.utils.util_classes import Scraper
from scripts.utils.util_classes import CustomLogger
import yfinance as yf
import pandas as pd

logger = CustomLogger.get_logger(name = __name__)

class yfinanceScraper(Scraper):
    def __init__(self, db, collection_name , start_data , end_data , tickers = None ,scraper_name = 'yfinance_scraper'):
        super().__init__(db, collection_name , scraper_name = scraper_name)
        self.name = 'yfinance'
        self.start_data = start_data
        self.end_data = end_data
        self.tickers = tickers

    def retrieve_data(self ):
        i = 0
        n_tickers = len(self.tickers)
        batch_data = []  # List to hold individual DataFrames for each ticker

        while i < n_tickers:
            ticker = self.tickers[i]
            logger.info(f'Downloading data for {ticker} ({i + 1}/{n_tickers}) ...')

            try:
                ticker_data = yf.download(ticker, start=self.start_data, end=self.end_data)
                ticker_data['ticker'] = ticker
                ticker_data = ticker_data.reset_index()
                ticker_data.columns = [col.lower() for col in ticker_data.columns]
                batch_data.append(ticker_data)  # Add this ticker's DataFrame to the batch

                # If we've reached a batch size of 50, yield the concatenated DataFrame
                if len(batch_data) == 200:
                    logger.info('Writing batch of 200 tickers to database')
                    yield pd.concat(batch_data)
                    batch_data = []  # Reset the list for the next batch

            except Exception:
                logger.info(f'Error downloading data for {ticker} ...')

            i += 1

        # Don't forget to yield any remaining data that didn't form a complete batch of 50
        if batch_data:
            logger.info('Writing last batch of tickers to database')
            yield pd.concat(batch_data)








if __name__ == "__main__":
    logger.info('Test')
    print('Done')
