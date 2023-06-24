
# Connection Setup
env_file_path = os.path.join(ROOT_DIR, 'env.yaml')

# Load environment variables from env.yaml
with open(env_file_path) as file:
    env_vars = yaml.safe_load(file)
username = env_vars.get('USER_NAME')
password = env_vars.get('PASS_WORD')

# Use the escaped username and password in the MongoDB connection string
mongo_db_url = f"mongodb+srv://{username}:{password}@rentalbike.5fi8zs7.mongodb.net/"

client = pymongo.MongoClient(mongo_db_url)
