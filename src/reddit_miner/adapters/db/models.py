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

    # ---- AI moderation fields ----
    ai_label = Column(Text)             # EXTREMIST / TOXIC / SAFE
    ai_explanation = Column(Text)       # text explaining why the label was chosen
    ai_model = Column(Text)             # model name or path used
    ai_timestamp = Column(DateTime)     # when the AI processed the comment
