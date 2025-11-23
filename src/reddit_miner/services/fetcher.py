# src/reddit_miner/services/fetcher.py

from typing import List, Optional
from praw import Reddit
from praw.models import Redditor
from prawcore.exceptions import TooManyRequests
from tqdm import tqdm
from colorama import Fore, Style
from datetime import datetime, timezone
from reddit_miner.domain.models import Comment
import json, os, time

# Directory to store user state
STATE_DIR = "src/reddit_miner/state"
os.makedirs(STATE_DIR, exist_ok=True)

def load_state(username: str) -> dict:
    """Load saved state for a Reddit user."""
    state_file = os.path.join(STATE_DIR, f"{username}.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {"after": None, "comments_fetched": 0}

def save_state(username: str, state: dict):
    """Save state for a Reddit user."""
    state_file = os.path.join(STATE_DIR, f"{username}.json")
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

def fetch_comments(reddit: Reddit, username: str, limit: Optional[int] = None) -> List[Comment]:
    """
    Fetch comments for a Reddit user using PRAW with automatic resume and rate-limit handling.

    Args:
        reddit: Initialized PRAW Reddit instance
        username: Reddit username
        limit: Maximum comments to fetch

    Returns:
        List[Comment]: List of Comment objects
    """
    user: Redditor = reddit.redditor(username)
    comments: List[Comment] = []

    # Load saved state
    state = load_state(username)
    comments_fetched = state.get("comments_fetched", 0)
    after_id = state.get("after")

    # Estimate total for tqdm
    estimated_total = limit if limit else 10000
    pbar = tqdm(total=estimated_total,
                desc=f"{Fore.YELLOW}Processing comments ({username}){Style.RESET_ALL}",
                initial=comments_fetched,
                unit="comment",
                ncols=80,
                colour="cyan")

    while True:
        try:
            batch = []
            for c in user.comments.new(limit=100):
                # Skip comments already fetched
                if after_id and c.id <= after_id:
                    continue
                batch.append(c)
                if limit and comments_fetched + len(batch) >= limit:
                    break

            if not batch:
                break

            for c in batch:
                # Safe parent fetch
                try:
                    parent_score = c.parent().score if c.parent() else 0
                except TooManyRequests:
                    parent_score = 0

                comment_obj = Comment(
                    comment_id=c.id,
                    author=c.author.name if c.author else "[deleted]",
                    subreddit=c.subreddit.display_name,
                    post_id=c.link_id.split("_")[1] if c.link_id else "",
                    post_title=c.link_title,
                    score=c.score,
                    num_comments=parent_score,
                    date_utc=datetime.fromtimestamp(c.created_utc, tz=timezone.utc),
                    permalink=f"https://reddit.com{c.permalink}",
                    body=c.body
                )
                comments.append(comment_obj)

                # Update state
                after_id = c.id
                comments_fetched += 1
                state["after"] = after_id
                state["comments_fetched"] = comments_fetched
                save_state(username, state)

                # Update progress bar
                pbar.update(1)

            if limit and comments_fetched >= limit:
                break

        except TooManyRequests as e:
            wait = int(getattr(e.response, "headers", {}).get("retry-after", 60))
            print(Fore.RED + f"\nRate limited by Reddit, sleeping {wait}s..." + Style.RESET_ALL)
            time.sleep(wait)
            # Reload state after sleep
            state = load_state(username)
            after_id = state.get("after")
            comments_fetched = state.get("comments_fetched", 0)
            continue

    pbar.close()
    return comments
