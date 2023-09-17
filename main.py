from scripts.utils.util_functions import read_config
from scripts.db.db_setup import mongodb_setup
from scripts.db.db_interface import dbInterface
from scripts.data_loader.tickers_loader import tickersScraper
from scripts.data_loader.yfinance_scraper import yfinanceScraper
from gui.app import run_app

import argparse
import ast


def run(args):
    # Import config
    config = read_config()
    tickers_collection_name = config['database']['collections']['tickers_collection_name']
    price_collection_name = config['database']['collections']['price_collection_name']
    batch_size = config['database']['batch_size']
    start_data = config['data']['start_data']
    end_data = config['data']['end_data']

    # Setup database
    if args.db == 'local' and args.db_initialise is True:
        mongodb_setup(config=config['database'] , local = True)
    if args.db == 'remote' and args.db_initialise is True:
        mongodb_setup(config=config['database'] , local = False)

    # Initialise database interface
    if args.db == 'local':
        db = dbInterface(config=config['database'], ticker_collection=tickers_collection_name,
                         price_collection=price_collection_name, batch_size=batch_size , local = True)
    elif args.db == 'remote':
        db = dbInterface(config=config['database'], ticker_collection=tickers_collection_name,
                         price_collection=price_collection_name, batch_size=batch_size , local = False)

    # Load tickers to database
    if args.db_initialise is True:
        tickers_scraper = tickersScraper(db=db, collection_name=tickers_collection_name )
        tickers_scraper.run()

        tickers = db.retrieve_tickers(indices=args.indices)[:20]
        price_scraper = yfinanceScraper(db=db, collection_name=price_collection_name, start_data=start_data,
                                        end_data=end_data, tickers=tickers )
        price_scraper.run()

    run_app(db)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify Database and Data Scraping Options")

    # Argument for specifying database type
    parser.add_argument('--db', default = 'remote', choices=['remote', 'local'],
                        help='Specify if you want to use a remote or local database.')


    def parse_bool(value):
        try:
            return ast.literal_eval(value.capitalize())
        except (ValueError, SyntaxError):
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Argument for enabling or disabling data scraping
    parser.add_argument('--db_initialise', type=parse_bool, default=False,
                        help='Initialise database with tickers and price data. True/False')

    # Add argument for specifying which indices to scrape
    parser.add_argument('--indices', nargs='+', default=['S&P 500' , 'Nasdaq 100' , 'Russell 2000'])

    args = parser.parse_args()

    run(args = args)
