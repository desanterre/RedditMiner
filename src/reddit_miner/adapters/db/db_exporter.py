from typing import List
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from reddit_miner.domain.models import Comment
from .models import CommentORM, Base

DB_URL = os.getenv("DATABASE_URL", "postgresql://reddit_user:reddit_pass@localhost:5432/reddit_db")

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

def export_to_postgres(comments: List[Comment]):
    if not comments:
        return

    orm_comments = [
        CommentORM(
            comment_id=c.comment_id,
            author=c.author,
            subreddit=c.subreddit,
            post_id=c.post_id,
            post_title=c.post_title,
            score=c.score,
            num_comments=c.num_comments,
            date_utc=c.date_utc,
            permalink=c.permalink,
            body=c.body
        ) for c in comments
    ]

    session = Session()
    try:
        session.add_all(orm_comments)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
