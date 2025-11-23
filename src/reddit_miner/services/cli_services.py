import os
from colorama import Fore
from collections import Counter
import praw

from reddit_miner.adapters.reddit.fetcher import fetch_comments
from reddit_miner.adapters.db.db_exporter import export_to_postgres, get_comments_for_user
from reddit_miner.services.exporters.excel_exporter import export_to_excel
from reddit_miner.services.user_simulator.user_simulator import UserSimulator
from reddit_miner.services.reddit_sentinel.reddit_sentinel import RedditSentinel

# -----------------------------
# Reddit Client
# -----------------------------
def get_reddit_client():
    """Initialize and return a PRAW Reddit client."""
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "reddit-miner")
    )

# -----------------------------
# Fetching Comments
# -----------------------------
def fetch_and_export_excel(usernames, limit=None, output=None):
    """Fetch comments for given users and export to Excel."""
    print(Fore.MAGENTA + f"Starting RedditMiner for users: {', '.join(usernames)}")
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
    """Fetch comments for users and export to PostgreSQL."""
    print(Fore.MAGENTA + f"Starting RedditMiner for users: {', '.join(usernames)}")
    reddit = get_reddit_client()

    all_comments = []
    for username in usernames:
        print(Fore.GREEN + f"\nFetching comments for user: {Fore.YELLOW}{username}{Fore.GREEN}")
        comments = fetch_comments(reddit, username, limit)
        all_comments.extend(comments)
        print(Fore.GREEN + f"Fetched {len(comments)} comments for {Fore.YELLOW}{username}")

    export_to_postgres(all_comments)
    print(Fore.MAGENTA + f"\nExported {len(all_comments)} comments to PostgreSQL")

# -----------------------------
# AI User Simulation
# -----------------------------
def simulate_user(username, model_path=None):
    """Simulate a Reddit user with GPT/AI using their comment history."""
    comments = get_comments_for_user(username)
    if comments.empty:
        print(Fore.RED + f"No comments found for user {username}")
        return

    analyzer = UserSimulator(model_path=model_path)
    analyzer.load_from_dataframe(comments)
    analyzer.load_model()
    analyzer.chat()

def reddit_sentinel_scan(output="flagged_comments.xlsx", model_path=None):
    """Scan comments for toxic/extremist content using AI."""
    sentinel = RedditSentinel(model_path=model_path)
    sentinel.load_comments_from_db()
    sentinel.load_model()
    sentinel.classify_comments()
    sentinel.export_flagged_comments(output)
    pass

# -----------------------------
# Subreddit Analytics
# -----------------------------
def get_top_users(subreddit, limit=10, comment_limit=2000):
    """Display the top users in a subreddit based on comment activity."""
    reddit = get_reddit_client()
    print(Fore.MAGENTA + f"\nFetching {comment_limit} comments from r/{subreddit}...")
    comments = [c.author.name for c in reddit.subreddit(subreddit).comments(limit=comment_limit) if c.author]

    if not comments:
        print(Fore.RED + "No comments found.")
        return

    counter = Counter(comments)
    top_users = counter.most_common(limit)
    print(Fore.GREEN + f"\nTop {limit} users in r/{subreddit} (based on last {comment_limit} comments):")
    print(Fore.CYAN + "-"*60)
    for rank, (username, count) in enumerate(top_users, start=1):
        print(f"{rank:2}. {username:<20} {count} comments")
    print(Fore.CYAN + "-"*60)

def get_top_posts(subreddit, limit=10):
    """Fetch top posts in a subreddit."""
    pass

def get_top_commenters_post(url, limit=10):
    """Fetch top commenters for a post URL."""
    pass

def get_user_top_subs(username, limit=10):
    """Show the top subreddits a user participates in."""
    pass

def get_user_hour_activity(username):
    """Analyze a user's comment activity by hour."""
    pass

def get_user_day_activity(username):
    """Analyze a user's comment activity by day of the week."""
    pass

def get_user_top_words(username, limit=20):
    """Show most frequent words used by a user."""
    pass

def get_user_comment_length_stats(username):
    """Provide stats on comment lengths (avg/min/max)."""
    pass

def extract_user_urls(username):
    """Extract all URLs shared by a user."""
    pass

def get_user_score_distribution(username):
    """Show distribution of comment scores for a user."""
    pass

def fetch_multi_subreddits(subs, limit=500):
    """Fetch posts/comments from multiple subreddits."""
    pass

def stream_subreddit(subreddit):
    """Stream subreddit comments in real-time."""
    pass

def search_keyword(subreddit, keyword, limit=500):
    """Search for a keyword in a subreddit's comments."""
    pass
