# Makefile for RedditMiner

.PHONY: install setup-model

install:
	@echo "Installing dependencies with poetry..."
	poetry install

setup-model:
	@echo "Creating GPT4All cache directory and copying model..."
	mkdir -p ~/.cache/gpt4all/models
	cp models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf ~/.cache/gpt4all/models/
	@echo "✓ GPT4All model ready in ~/.cache/gpt4all/models/"
