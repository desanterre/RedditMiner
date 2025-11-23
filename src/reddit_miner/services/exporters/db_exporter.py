from reddit_miner.services.fetcher import fetch_comments
from reddit_miner.adapters.db.db_exporter import export_to_postgres

def fetch_and_store(reddit, username, limit=None):
    comments = fetch_comments(reddit, username, limit)
    export_to_postgres(comments)
