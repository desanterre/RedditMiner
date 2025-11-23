import click
import os
from colorama import init, Fore
from reddit_miner.services.cli_services import get_reddit_client, print_startup_info
from reddit_miner.adapters.reddit.fetcher import fetch_comments
from reddit_miner.services.exporters.excel_exporter import export_to_excel
from reddit_miner.adapters.db.db_exporter import export_to_postgres
from reddit_miner.services.user_simulator.analyzer import analyze_and_chat

init(autoreset=True)

@click.group()
def main():
    """RedditMiner CLI - Fetch Reddit comments and analyze with AI"""
    pass


# ----------------------------
# Excel export command
# ----------------------------
@main.command()
@click.option("--usernames", "-u", multiple=True, required=True, help="Reddit usernames")
@click.option("--limit", "-l", type=int, default=None, help="Max number of comments per user")
@click.option("--output", "-o", type=str, default=None, help="Output Excel filename")
def fetch_excel(usernames, limit, output):
    """Fetch Reddit user comments and export to Excel."""
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


# ----------------------------
# PostgreSQL export command
# ----------------------------
@main.command()
@click.option("--usernames", "-u", multiple=True, required=True, help="Reddit usernames")
@click.option("--limit", "-l", type=int, default=None, help="Max number of comments per user")
def fetch_db(usernames, limit):
    """Fetch Reddit user comments and export to PostgreSQL."""
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


# ----------------------------
# Chat/analysis command
# ----------------------------
@main.command()
@click.option("--file", "-f", type=str, required=True, help="Excel file to analyze")
def chat(file):
    """Analyze Reddit user comments and start chatbot simulation."""
    if not os.path.exists(file):
        print(f"{Fore.RED}File not found: {file}")
        return
    analyze_and_chat(file)


if __name__ == "__main__":
    main()
