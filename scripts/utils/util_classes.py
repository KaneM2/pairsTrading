from scripts.utils.util_functions import read_config
from config import BASE_DIR

import logging
import os
import datetime as dt


class CustomLogger:
    @classmethod
    def get_logger(cls, name):
        config = read_config('log')
        logger = logging.getLogger(name)

        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

        file_handler = logging.FileHandler(os.path.join(BASE_DIR , 'log.txt'))
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

logger = CustomLogger.get_logger(name = __name__)
class Scraper:
    def __init__(self , db  ,collection_name ,scraper_name = 'base'):
        self.db = db
        self.collection_name = collection_name
        self.scraper_name = scraper_name

    def retrieve_data(self):
        raise NotImplementedError

    def write_to_db(self , data):
        self.db.write_dataframe(df = data , collection_name= self.collection_name )


    def run(self):
        t1 = dt.datetime.now()
        data_generator = self.retrieve_data()
        logger.info('Writing data to {} ...'.format(self.collection_name))
        for data in data_generator:
            self.write_to_db(data)
        t2 = dt.datetime.now()
        logger.info('{} completed in {} seconds'.format( self.scraper_name , (t2-t1).seconds))


