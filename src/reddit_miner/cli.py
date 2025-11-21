# src/reddit_miner/cli.py

import click
import praw, os
from reddit_miner.services.fetcher import fetch_comments
from reddit_miner.services.exporter import export_to_excel
from colorama import init, Fore, Style
from datetime import datetime

init(autoreset=True)  # automatically reset colors after each print

@click.command()
@click.option("--usernames", "-u", multiple=True, required=True, help="Reddit usernames")
@click.option("--limit", "-l", type=int, default=None, help="Max number of comments per user")
@click.option("--output", "-o", type=str, default=None, help="Output Excel filename")
def main(usernames, limit, output):
    """
    CLI entry point for RedditMiner: fetch Reddit user comments and export to Excel.
    """
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

    # Display configuration info
    print(Fore.CYAN + f"RedditMiner v0.1.0")
    print(Fore.CYAN + f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(Fore.CYAN + f"Using Reddit App: {Fore.YELLOW}{client_id}{Fore.CYAN}")
    print(Fore.CYAN + f"Usernames: {', '.join(usernames)}\n")

    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="RedditMiner/1.0"
    )

    all_comments = []

    for username in usernames:
        print(Fore.GREEN + f"\nFetching comments for user: {Fore.YELLOW}{username}{Fore.GREEN}")
        # Pass reddit instance and username string to fetch_comments
        comments = fetch_comments(reddit, username, limit)
        all_comments.extend(comments)
        print(Fore.GREEN + f"Fetched {len(comments)} comments for {Fore.YELLOW}{username}")

    # Determine output filename
    filename = output or f"reddit_comments_{'_'.join(usernames)}.xlsx"
    export_to_excel(all_comments, filename)
    print(Fore.MAGENTA + f"\nSaved comments to {filename}")


if __name__ == "__main__":
    main()
