from typing import List, Optional
from praw import Reddit
from tqdm import tqdm
from colorama import Fore, Style
from datetime import datetime, timezone
from reddit_miner.domain.models import Comment

def fetch_comments(reddit: Reddit, username: str, limit: Optional[int] = None) -> List[Comment]:
    """
    Fetch comments for a Reddit user using PRAW.

    Args:
        reddit: Initialized PRAW Reddit instance
        username: Reddit username
        limit: Maximum comments to fetch

    Returns:
        List[Comment]: List of comment domain objects
    """
    user = reddit.redditor(username)
    comments: List[Comment] = []

    for c in tqdm(user.comments.new(limit=limit),
                  desc=f"{Fore.YELLOW}Processing comments",
                  unit="comment",
                  ncols=80,
                  colour="cyan"):
        comments.append(Comment(
            comment_id=c.id,
            author=c.author.name if c.author else "[deleted]",
            subreddit=c.subreddit.display_name,
            post_id=c.link_id.split("_")[1],
            post_title=c.link_title,
            score=c.score,
            num_comments=c.parent().score if c.parent() else 0,
            date_utc=datetime.fromtimestamp(c.created_utc, tz=timezone.utc),
            permalink=f"https://reddit.com{c.permalink}",
            body=c.body
        ))

    print(f"{Fore.MAGENTA}Fetched {len(comments)} comments for {username}{Style.RESET_ALL}")
    return comments
