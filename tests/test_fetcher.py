import pytest
from unittest.mock import MagicMock
from reddit_miner.services.fetcher import fetch_comments
from reddit_miner.domain.models import Comment

def test_fetch_comments():
    reddit_mock = MagicMock()
    user_mock = MagicMock()
    reddit_mock.redditor.return_value = user_mock

    comment_mock = MagicMock()
    comment_mock.id = "abc123"
    comment_mock.author.name = "testuser"
    comment_mock.subreddit.display_name = "testsub"
    comment_mock.link_id = "t3_postid"
    comment_mock.link_title = "Post Title"
    comment_mock.score = 10
    comment_mock.parent.return_value.score = 5
    comment_mock.created_utc = 1690000000
    comment_mock.permalink = "/r/testsub/comments/postid/commentid"
    comment_mock.body = "Test comment"

    user_mock.comments.new.return_value = [comment_mock]

    comments = fetch_comments(reddit_mock, "testuser", limit=1)
    assert isinstance(comments[0], Comment)
    assert comments[0].author == "testuser"
