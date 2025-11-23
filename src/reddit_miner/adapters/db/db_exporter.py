import os
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from .models import CommentORM, Base

DB_URL = os.getenv("DATABASE_URL", "postgresql://reddit_user:reddit_pass@localhost:5432/reddit_db")

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

def export_to_postgres(comments):
    if not comments:
        return

    orm_comments = [{
        "comment_id": c.comment_id,
        "author": c.author,
        "subreddit": c.subreddit,
        "post_id": c.post_id,
        "post_title": c.post_title,
        "score": c.score,
        "num_comments": c.num_comments,
        "date_utc": c.date_utc,
        "permalink": c.permalink,
        "body": c.body
    } for c in comments]

    session = Session()
    try:
        stmt = insert(CommentORM).values(orm_comments)
        stmt = stmt.on_conflict_do_nothing(index_elements=["comment_id"])
        session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_comments_for_user(username: str) -> pd.DataFrame:
    session = Session()
    try:
        query = session.query(CommentORM).filter(CommentORM.author == username)
        results = query.all()
        df = pd.DataFrame([{
            "Author": c.author,
            "Subreddit": c.subreddit,
            "PostID": c.post_id,
            "PostTitle": c.post_title,
            "Score": c.score,
            "NumComments": c.num_comments,
            "DateUTC": c.date_utc,
            "Permalink": c.permalink,
            "Body": c.body
        } for c in results])
        return df
    finally:
        session.close()

def get_all_comments() -> pd.DataFrame:
    """
    Fetch all comments from the database as a DataFrame.
    """
    session = Session()
    try:
        results = session.query(CommentORM).all()
        df = pd.DataFrame([{
            "Author": c.author,
            "Subreddit": c.subreddit,
            "PostID": c.post_id,
            "PostTitle": c.post_title,
            "Score": c.score,
            "NumComments": c.num_comments,
            "DateUTC": c.date_utc,
            "Permalink": c.permalink,
            "Body": c.body
        } for c in results])
        return df
    finally:
        session.close()
