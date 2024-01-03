
init:
    echo "TOKEN=test" >> .env
up:
    docker compose build nb && docker compose up -d nb
