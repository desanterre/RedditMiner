import click
import os
from colorama import init, Fore
from reddit_miner.services.cli_services import (
    fetch_and_export_excel,
    fetch_and_export_db,
    simulate_user
)

init(autoreset=True)

@click.group()
def main():
    """RedditMiner CLI - Fetch Reddit comments and analyze with AI"""
    pass

@main.command()
@click.option("--usernames", "-u", multiple=True, required=True, help="Reddit usernames")
@click.option("--limit", "-l", type=int, default=None, help="Max number of comments per user")
@click.option("--output", "-o", type=str, default=None, help="Output Excel filename")
def fetch_excel(usernames, limit, output):
    fetch_and_export_excel(usernames, limit, output)

@main.command()
@click.option("--usernames", "-u", multiple=True, required=True, help="Reddit usernames")
@click.option("--limit", "-l", type=int, default=None, help="Max number of comments per user")
def fetch_db(usernames, limit):
    fetch_and_export_db(usernames, limit)

@main.command()
@click.option("--username", "-u", type=str, required=True, help="Reddit username to simulate")
@click.option("--model", "-m", type=str, default="models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf", help="GPT4All model path")
def chat(username, model):
    simulate_user(username, model)

if __name__ == "__main__":
    main()
