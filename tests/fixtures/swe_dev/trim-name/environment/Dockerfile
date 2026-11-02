# Fixture build example only; production manifests require a resolved image digest.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git tmux asciinema && rm -rf /var/lib/apt/lists/*
WORKDIR /app/repo
COPY repo/ /app/repo/
