from pymongo import MongoClient
from scripts.utils.util_functions import read_config
from scripts.utils.util_classes import CustomLogger

logger = CustomLogger.get_logger(name = __name__)
def mongodb_setup(config = read_config('database') , local = True):
    logger.info('Initialising database ...')
    if local is True:
        credentials_config = config['local']
        host = credentials_config['host']
        port = credentials_config['port']
        client = MongoClient('mongodb://{}:{}/'.format(host , port))
    elif local is False:
        credentials_config = config['remote']
        uri = ("mongodb+srv://" + credentials_config['user'] + ":" + credentials_config['password'] + "@"
               + credentials_config['host'] + "/?retryWrites=true&w=majority")
        client = MongoClient(uri)
    else:
        raise NotImplementedError

    name = credentials_config['db_name']
    db = client[name]

    logger.info('Creating database collections ...')
    for collection_name in config['collections']:
        new_collection = db[config['collections'][collection_name]]

    logger.info('Creating database collection indices')
    for collection_conf in config['indices']:
        for collection, fields_str in collection_conf.items():
            fields_tuple = eval(fields_str)
            collection = db[collection]

            try:
                collection.create_index(list(fields_tuple) , unique = True)

            except Exception as e:
                logger.debug(e)


