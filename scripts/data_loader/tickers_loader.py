from scripts.utils.util_classes import CustomLogger
from scripts.utils.util_classes import Scraper
from config import BASE_DIR
import pandas as pd
import os

logger = CustomLogger.get_logger(name = __name__)

class tickersScraper(Scraper):
    def __init__(self, db, collection_name , scraper_name = 'tickers_scraper'):
        super().__init__(db, collection_name , scraper_name = scraper_name)


    def retrieve_data(self  ):
        logger.info('Loading ticker information from csv ...')
        tickers = pd.read_csv(os.path.join(BASE_DIR , 'data/tickers_meta.csv'))
        yield tickers


