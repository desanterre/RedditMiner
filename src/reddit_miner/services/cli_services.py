import os
import praw
from colorama import Fore
from datetime import datetime

def get_reddit_client():
    """Return a PRAW Reddit instance using environment variables."""
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in your environment")

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="RedditMiner/1.0"
    )


def print_startup_info(usernames):
    """Print info at the start of the CLI command."""
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    print(Fore.CYAN + f"RedditMiner v0.1.0")
    print(Fore.CYAN + f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(Fore.CYAN + f"Using Reddit App: {Fore.YELLOW}{client_id}{Fore.CYAN}")
    print(Fore.CYAN + f"Usernames: {', '.join(usernames)}\n")
