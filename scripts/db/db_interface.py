from pymongo import MongoClient , errors
import pandas as pd
from scripts.utils.util_classes import CustomLogger

logger = CustomLogger.get_logger(name = __name__)

class dbInterface():
    def __init__(self , config , ticker_collection  , price_collection , local = True, batch_size = 10000):
        if local is True:
            credentials_config = config['local']
            host = credentials_config['host']
            port = credentials_config['port']
            self.client = MongoClient(host = host, port = port)
        elif local is False:
            credentials_config = config['remote']
            uri = ("mongodb+srv://" + credentials_config['user'] + ":" + credentials_config['password'] + "@"
                   + credentials_config['host'] + "/?retryWrites=true&w=majority")
            self.client = MongoClient(uri)

        self.db = self.client[credentials_config['db_name']]
        self.ticker_collection = ticker_collection
        self.price_collection = price_collection
        self.batch_size = batch_size

    def write_dataframe(self , df , collection_name):
        records = df.to_dict('records')
        collection = self.db[collection_name]
        for i in range(0, len(records), self.batch_size):
            batch = records[i: i + self.batch_size]
            try:
                collection.insert_many(batch, ordered = False)
            except errors.BulkWriteError as bwe:
                logger.debug('Some indices are duplicated , skipping duplicate keys')


    def retrive_sectors(self):
        return self.db[self.ticker_collection].distinct('sector')

    def retrieve_indices(self):
        return self.db[self.ticker_collection].distinct('index')

    def retrieve_tickers(self, sectors=None, indices=None):

        query = {}

        if sectors is not None:
            query["sector"] = {"$in": sectors}

        if indices is not None:
            query["index"] = {"$in": indices}

        cursor = self.db[self.ticker_collection].find(query, {"ticker": 1, "_id": 0})

        # Create a DataFrame from the cursor
        df = pd.DataFrame(list(cursor))

        # Remove duplicate tickers if needed
        df.drop_duplicates(subset=["ticker"], inplace=True)
        # df to list
        tickers = df["ticker"].tolist()
        return tickers

    def retrieve_prices(self, tickers=None, start_date=None, end_date=None):

            query = {}

            if tickers is not None:
                query["ticker"] = {"$in": tickers}

            date_query = {}

            if start_date is not None:
                date_query["$gte"] =  pd.to_datetime(start_date)

            if end_date is not None:
                date_query["$lte"] = pd.to_datetime(end_date)

            if date_query != {}:
                query["date"] = date_query

            cursor = self.db[self.price_collection].find(query)

            # Create a DataFrame from the cursor
            df = pd.DataFrame(list(cursor)).drop(columns = ['_id']).set_index(['date' , 'ticker'])
            cursor.close()

            return df

    def retrieve_saved_pair_names(self):
        return self.db['pairsCollection'].distinct('nametag')

    def retrieve_pairs(self , nametag):
        query = {'nametag' : nametag}
        return self.db['pairsCollection'].find(query)[0]['pairs']







if __name__ == '__main__':
    print('Done')

