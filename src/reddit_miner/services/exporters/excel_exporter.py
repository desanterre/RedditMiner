from typing import List, Optional
import pandas as pd
from reddit_miner.domain.models import Comment
from datetime import datetime
import os

def export_to_excel(comments: List[Comment], filename: Optional[str] = None) -> str:
    """
    Export a list of Comment objects to an Excel file.

    Args:
        comments (List[Comment]): List of comment domain objects
        filename (Optional[str]): Output Excel filename. If None, generates a timestamped filename.

    Returns:
        str: The filename of the exported Excel file
    """
    if not filename:
        filename = f"reddit_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    # Convert comments to dicts for pandas
    data = [
        {
            "CommentID": c.comment_id,
            "Author": c.author,
            "Subreddit": c.subreddit,
            "PostID": c.post_id,
            "PostTitle": c.post_title,
            "Score": c.score,
            "NumComments": c.num_comments,
            # If c.date_utc is already a datetime, format it directly
            "DateUTC": c.date_utc.strftime("%Y-%m-%d %H:%M:%S") if isinstance(c.date_utc, datetime) else datetime.utcfromtimestamp(c.date_utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Permalink": c.permalink,
            "Body": c.body,
        } for c in comments
    ]

    # Append to existing Excel if it exists
    if os.path.exists(filename):
        df_existing = pd.read_excel(filename)
        df = pd.concat([df_existing, pd.DataFrame(data)], ignore_index=True)
    else:
        df = pd.DataFrame(data)

    # Export to Excel
    df.to_excel(filename, index=False)
    return filename
