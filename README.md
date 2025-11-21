# RedditMiner

CLI tool to scrape Reddit user comments and export them to Excel with color logging and progress bar.

## Installation

```bash
git clone <repo-url>
cd RedditMiner
poetry install
```

## Usage

```bash
poetry run reddit-miner --usernames spez --limit 50
```

``` Options

- `--usernames` / `-u`: List of Reddit usernames (space-separated)  
- `--limit` / `-l`: Max number of comments per user (optional)  
- `--output` / `-o`: Output Excel filename (optional)  

``` Environment Variables

- `REDDIT_CLIENT_ID` : Reddit API client ID  
- `REDDIT_CLIENT_SECRET` : Reddit API client secret  

---

## 🔐 How to get a Reddit OAuth Token

1. Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)  
2. Create a new **script application**  
3. Copy the **Client ID** and **Client Secret**  
4. Set environment variables:

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
```

5. Run RedditMiner:

```bash
poetry run reddit-miner -u spez -l 100
```
