import os
from colorama import Fore
from reddit_miner.adapters.reddit.fetcher import fetch_comments
from reddit_miner.adapters.db.db_exporter import export_to_postgres, get_comments_for_user
from reddit_miner.services.exporters.excel_exporter import export_to_excel
from reddit_miner.services.user_simulator.user_simulator import UserSimulator
import praw

def get_reddit_client():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "reddit-miner")
    )

def print_startup_info(usernames):
    print(Fore.MAGENTA + f"Starting RedditMiner for users: {', '.join(usernames)}")

def fetch_and_export_excel(usernames, limit=None, output=None):
    print_startup_info(usernames)
    reddit = get_reddit_client()

    all_comments = []
    for username in usernames:
        print(Fore.GREEN + f"\nFetching comments for user: {Fore.YELLOW}{username}{Fore.GREEN}")
        comments = fetch_comments(reddit, username, limit)
        all_comments.extend(comments)
        print(Fore.GREEN + f"Fetched {len(comments)} comments for {Fore.YELLOW}{username}")

    filename = output or f"reddit_comments_{'_'.join(usernames)}.xlsx"
    export_to_excel(all_comments, filename)
    print(Fore.MAGENTA + f"\nSaved {len(all_comments)} comments to {filename}")

def fetch_and_export_db(usernames, limit=None):
    print_startup_info(usernames)
    reddit = get_reddit_client()

    all_comments = []
    for username in usernames:
        print(Fore.GREEN + f"\nFetching comments for user: {Fore.YELLOW}{username}{Fore.GREEN}")
        comments = fetch_comments(reddit, username, limit)
        all_comments.extend(comments)
        print(Fore.GREEN + f"Fetched {len(comments)} comments for {Fore.YELLOW}{username}")

    export_to_postgres(all_comments)
    print(Fore.MAGENTA + f"\nExported {len(all_comments)} comments to PostgreSQL")

def simulate_user(username, model_path=None):
    comments = get_comments_for_user(username)  # returns DataFrame
    if comments.empty:
        print(Fore.RED + f"No comments found for user {username}")
        return

    analyzer = UserSimulator(model_path=model_path)
    analyzer.load_from_dataframe(comments)
    analyzer.load_model()
    analyzer.chat()

