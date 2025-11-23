from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Text, Integer, DateTime

Base = declarative_base()

class CommentORM(Base):
    __tablename__ = "comments"

    comment_id = Column(Text, primary_key=True)
    author = Column(Text)
    subreddit = Column(Text)
    post_id = Column(Text)
    post_title = Column(Text)
    score = Column(Integer)
    num_comments = Column(Integer)
    date_utc = Column(DateTime)
    permalink = Column(Text)
    body = Column(Text)
