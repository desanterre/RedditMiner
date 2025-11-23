# reddit_miner/cli.py
import click
from colorama import init, Fore

from reddit_miner.services.cli_services import (
    fetch_and_export_excel,
    fetch_and_export_db,
    simulate_user,
    get_top_users,
    get_top_posts,
    get_top_commenters_post,
    get_user_top_subs,
    get_user_hour_activity,
    get_user_day_activity,
    get_user_top_words,
    get_user_comment_length_stats,
    extract_user_urls,
    get_user_score_distribution,
    fetch_multi_subreddits,
    stream_subreddit,
    search_keyword,
    reddit_sentinel_scan,
)

init(autoreset=True)

@click.group()
def main():
    """RedditMiner CLI — Advanced Reddit scraping & AI user simulation"""
    pass

# Helper decorator for self-documenting commands
def documented_command(purpose: str, example: str):
    def decorator(f):
        f.__doc__ = f"{purpose}\n\nExample usage:\n{example}"
        return f
    return decorator


# ---------------------------------------------------------
# Fetching Commands
# ---------------------------------------------------------

@main.command()
@documented_command(
    purpose="Fetch Reddit comments for one or more users and export them to an Excel file.",
    example="poetry run reddit-miner fetch-excel -u spez -u kn0thing -l 500 -o output.xlsx"
)
@click.option("--usernames", "-u", multiple=True, required=True, help="Reddit usernames to fetch comments from")
@click.option("--limit", "-l", type=int, default=None, help="Maximum number of comments per user")
@click.option("--output", "-o", type=str, default=None, help="Output Excel filename")
def fetch_excel(usernames, limit, output):
    fetch_and_export_excel(usernames, limit, output)


@main.command()
@documented_command(
    purpose="Fetch Reddit comments for one or more users and store them in PostgreSQL database.",
    example="poetry run reddit-miner fetch-db -u spez -u kn0thing -l 1000"
)
@click.option("--usernames", "-u", multiple=True, required=True, help="Reddit usernames to fetch comments from")
@click.option("--limit", "-l", type=int, default=None, help="Maximum number of comments per user")
def fetch_db(usernames, limit):
    fetch_and_export_db(usernames, limit)


# ---------------------------------------------------------
# AI User Simulation
# ---------------------------------------------------------

@main.command()
@documented_command(
    purpose="Simulate a Reddit user using GPT/AI. Clone the user’s comment history and let the AI chat like them.",
    example="poetry run reddit-miner chat -u spez -m models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"
)
@click.option("--username", "-u", type=str, required=True, help="Reddit username to simulate")
@click.option(
    "--model", "-m",
    type=str,
    default="models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf",
    help="Path to GPT4All model for simulation"
)
def chat(username, model):
    simulate_user(username, model)


# ---------------------------------------------------------
# Subreddit Analytics
# ---------------------------------------------------------

@main.command()
@documented_command(
    purpose="Show top users in a subreddit based on the number of comments fetched.",
    example="poetry run reddit-miner top-users -s france -l 10 -c 2000"
)
@click.option("--subreddit", "-s", type=str, required=True, help="Subreddit name to analyze")
@click.option("--limit", "-l", type=int, default=10, help="Number of top users to show")
@click.option("--comment-limit", "-c", type=int, default=2000, help="Number of recent comments to fetch from subreddit")
def top_users(subreddit, limit, comment_limit):
    get_top_users(subreddit, limit, comment_limit)


@main.command()
@documented_command(
    purpose="Display the top posts in a subreddit by score or engagement.",
    example="poetry run reddit-miner top-posts -s france -l 20"
)
@click.option("--subreddit", "-s", type=str, required=True, help="Subreddit name to analyze")
@click.option("--limit", "-l", type=int, default=10, help="Number of top posts to fetch")
def top_posts(subreddit, limit):
    get_top_posts(subreddit, limit)


@main.command()
@documented_command(
    purpose="Get the top commenters on a specific Reddit post.",
    example="poetry run reddit-miner top-commenters-post -u https://reddit.com/r/france/comments/abcd1234/post-title -l 10"
)
@click.option("--url", "-u", type=str, required=True, help="Full URL of the Reddit post")
@click.option("--limit", "-l", type=int, default=10, help="Number of top commenters to display")
def top_commenters_post(url, limit):
    get_top_commenters_post(url, limit)


@main.command()
@documented_command(
    purpose="Fetch posts from multiple subreddits and aggregate them in one dataset.",
    example="poetry run reddit-miner multi-subs -s france -s worldnews -l 500"
)
@click.option("--subs", "-s", multiple=True, required=True, help="List of subreddit names to fetch")
@click.option("--limit", "-l", type=int, default=500, help="Max number of posts/comments per subreddit")
def multi_subs(subs, limit):
    fetch_multi_subreddits(list(subs), limit)


@main.command()
@documented_command(
    purpose="Stream new comments from a subreddit in real-time.",
    example="poetry run reddit-miner stream -s france"
)
@click.option("--subreddit", "-s", type=str, required=True, help="Subreddit name to stream")
def stream(subreddit):
    stream_subreddit(subreddit)


# ---------------------------------------------------------
# User Analytics
# ---------------------------------------------------------

@main.command()
@documented_command(
    purpose="Show the top subreddits a user participates in based on comment volume.",
    example="poetry run reddit-miner user-top-subs -u spez -l 10"
)
@click.option("--username", "-u", type=str, required=True, help="Reddit username")
@click.option("--limit", "-l", type=int, default=10, help="Number of top subreddits to show")
def user_top_subs(username, limit):
    get_user_top_subs(username, limit)


@main.command()
@documented_command(
    purpose="Analyze a user’s comment activity by hour of day.",
    example="poetry run reddit-miner user-hour-activity -u spez"
)
@click.option("--username", "-u", type=str, required=True, help="Reddit username")
def user_hour_activity(username):
    get_user_hour_activity(username)


@main.command()
@documented_command(
    purpose="Analyze a user’s comment activity by day of the week.",
    example="poetry run reddit-miner user-day-activity -u spez"
)
@click.option("--username", "-u", type=str, required=True, help="Reddit username")
def user_day_activity(username):
    get_user_day_activity(username)


@main.command()
@documented_command(
    purpose="Show the most frequent words used by a user in comments.",
    example="poetry run reddit-miner user-top-words -u spez -l 20"
)
@click.option("--username", "-u", type=str, required=True, help="Reddit username")
@click.option("--limit", "-l", type=int, default=20, help="Number of top words to display")
def user_top_words(username, limit):
    get_user_top_words(username, limit)


@main.command()
@documented_command(
    purpose="Provide statistics on user comment lengths (avg, min, max).",
    example="poetry run reddit-miner user-comment-stats -u spez"
)
@click.option("--username", "-u", type=str, required=True, help="Reddit username")
def user_comment_stats(username):
    get_user_comment_length_stats(username)


@main.command()
@documented_command(
    purpose="Extract all URLs shared by a user in comments.",
    example="poetry run reddit-miner user-urls -u spez"
)
@click.option("--username", "-u", type=str, required=True, help="Reddit username")
def user_urls(username):
    extract_user_urls(username)


@main.command()
@documented_command(
    purpose="Show the distribution of comment scores for a user.",
    example="poetry run reddit-miner user-score-distribution -u spez"
)
@click.option("--username", "-u", type=str, required=True, help="Reddit username")
def user_score_distribution(username):
    get_user_score_distribution(username)


# ---------------------------------------------------------
# Keyword Search
# ---------------------------------------------------------

@main.command()
@documented_command(
    purpose="Search for a keyword in a subreddit’s recent comments.",
    example="poetry run reddit-miner search -s france -k 'Macron' -l 500"
)
@click.option("--subreddit", "-s", type=str, required=True, help="Subreddit to search in")
@click.option("--keyword", "-k", type=str, required=True, help="Keyword to search for")
@click.option("--limit", "-l", type=int, default=500, help="Number of comments to scan")
def search(subreddit, keyword, limit):
    search_keyword(subreddit, keyword, limit)


# ---------------------------------------------------------
# RedditSentinel — AI moderation
# ---------------------------------------------------------

@main.command()
@documented_command(
    purpose="Scan comments in the database for extremist, racist, homophobic, or toxic content using AI and export to Excel.",
    example="poetry run reddit-miner reddit-sentinel -o flagged_comments.xlsx -m models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"
)
@click.option("--output", "-o", type=str, default="flagged_comments.xlsx", help="Excel file to save flagged comments")
@click.option(
    "--model", "-m",
    type=str,
    default="models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf",
    help="Path to GPT4All model for AI classification"
)
def reddit_sentinel(output, model):
    """Run AI scan for extremist/toxic content and export results"""
    reddit_sentinel_scan(output=output, model_path=model)

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
