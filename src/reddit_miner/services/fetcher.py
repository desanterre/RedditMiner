from typing import List, Optional
from praw import Reddit
from praw.models import Redditor
from prawcore.exceptions import TooManyRequests, NotFound, Forbidden
from colorama import Fore, Style, init
from datetime import datetime, timezone
from reddit_miner.domain.models import Comment
import json, os, time, sys

init(autoreset=True)

def load_state(username: str) -> dict:
    """Load saved state for a Reddit user."""
    state_file = os.path.join("src/reddit_miner/state", f"{username}.json")
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {"after": None, "comments_fetched": 0}

def save_state(username: str, state: dict):
    """Save state for a Reddit user."""
    state_file = os.path.join("src/reddit_miner/state", f"{username}.json")
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

def _print_progress_bar(current: int, total: int, username: str, start_time: float):
    """Print a stylized single progress bar on one line (update every 10 items)."""
    elapsed = time.time() - start_time
    if current > 0:
        rate = current / elapsed
        remaining = (total - current) / rate if rate > 0 else 0
    else:
        rate = 0
        remaining = 0

    percent = (current / total) if total > 0 else 0
    bar_length = 40
    filled = int(bar_length * percent)

    # Gradient colors
    if percent < 0.33:
        bar_color = Fore.CYAN
    elif percent < 0.66:
        bar_color = Fore.YELLOW
    else:
        bar_color = Fore.GREEN

    bar = "█" * filled + "░" * (bar_length - filled)

    time_str = f"{int(elapsed)}s"
    remaining_str = f"{int(remaining)}s" if remaining > 0 else "0s"
    rate_str = f"{rate:.1f}/s"

    progress_line = (
        f"{Fore.MAGENTA}▶ {Fore.CYAN}{username:15s} "
        f"{bar_color}│{bar}│ "
        f"{Fore.WHITE}{current:4d}/{total:4d} "
        f"{Fore.YELLOW}{percent*100:5.1f}% "
        f"{Fore.WHITE}⏱ {time_str:4s} "
        f"{Fore.LIGHTBLACK_EX}ETA {remaining_str:4s} "
        f"{Fore.GREEN}⚡ {rate_str:6s}"
        f"{Style.RESET_ALL}"
    )

    sys.stdout.write("\r" + progress_line)
    sys.stdout.flush()

def fetch_comments(reddit: Reddit, username: str, limit: Optional[int] = None, max_workers: int = 4) -> List[Comment]:
    """
    Fetch comments for a Reddit user — FAST version without parent score lookups.

    Args:
        reddit: Initialized PRAW Reddit instance
        username: Reddit username
        limit: Maximum comments to fetch (default 1000)
        max_workers: Ignored (kept for compatibility)

    Returns:
        List[Comment]: List of Comment objects
    """
    user: Redditor = reddit.redditor(username)
    comments: List[Comment] = []

    # Load saved state
    state = load_state(username)

    if limit is None:
        limit = 1000

    start_time = time.time()
    update_frequency = 10  # Update progress bar every 10 comments

    try:
        # Fetch user comments directly without parent lookup (FAST!)
        for c in user.comments.new(limit=limit):
            try:
                comment = Comment(
                    comment_id=c.id,
                    author=c.author.name if c.author else "[deleted]",
                    subreddit=c.subreddit.display_name,
                    post_id=c.link_id.split("_")[1] if c.link_id else "",
                    post_title=c.link_title,
                    score=c.score,
                    num_comments=0,
                    date_utc=datetime.fromtimestamp(c.created_utc, tz=timezone.utc),
                    permalink=f"https://reddit.com{c.permalink}",
                    body=c.body
                )
                comments.append(comment)

                # Update progress bar only every N comments
                if len(comments) % update_frequency == 0:
                    _print_progress_bar(len(comments), limit, username, start_time)

            except Exception as e:
                print(f"\n{Fore.RED}✗ Error processing comment: {e}{Style.RESET_ALL}")
                continue

    except TooManyRequests as e:
        wait = int(getattr(e.response, "headers", {}).get("retry-after", 60))
        print(f"\n{Fore.RED}⚠ Rate limited! Sleeping {wait}s...{Style.RESET_ALL}")
        time.sleep(wait)
    except (NotFound, Forbidden):
        print(f"\n{Fore.RED}✗ User '{username}' not found or unavailable (404/403). Skipping.{Style.RESET_ALL}")
        return []

    # Final update
    _print_progress_bar(len(comments), limit, username, start_time)
    print()  # New line after progress bar

    # Save final state
    if comments:
        state["after"] = comments[-1].comment_id
        state["comments_fetched"] = len(comments)
        save_state(username, state)

    return comments