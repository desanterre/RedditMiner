DB_CONTAINER_NAME=reddit_pg
DB_IMAGE_NAME=reddit_pg_image
PYTHON_IMAGE_NAME=reddit_py

.PHONY: all build_db start_db stop_db migrate install setup-model

all: build_db start_db migrate

# Build the PostgreSQL Docker image
build_db:
	docker build -t $(DB_IMAGE_NAME) src/reddit_miner/adapters/db/docker

# Start PostgreSQL container
start_db:
	docker run -d --name $(DB_CONTAINER_NAME) -p 5432:5432 $(DB_IMAGE_NAME)

# Stop and remove PostgreSQL container
stop_db:
	docker stop $(DB_CONTAINER_NAME) || true
	docker rm $(DB_CONTAINER_NAME) || true

# Run Alembic migrations
migrate:
	poetry run alembic -c alembic.ini upgrade head

# Install Python dependencies via poetry
install:
	@echo "Installing dependencies with poetry..."
	poetry install

# Setup GPT4All model cache
setup-model:
	@echo "Creating GPT4All cache directory and copying model..."
	mkdir -p ~/.cache/gpt4all/models
	cp models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf ~/.cache/gpt4all/models/
	@echo "✓ GPT4All model ready in ~/.cache/gpt4all/models/"
