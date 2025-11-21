from dataclasses import dataclass

@dataclass
class Comment:
    comment_id: str
    author: str
    subreddit: str
    post_id: str
    post_title: str
    score: int
    num_comments: int
    date_utc: float
    permalink: str
    body: str
