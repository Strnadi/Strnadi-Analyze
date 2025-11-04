#!/bin/bash

git reset --hard
git pull --rebase
docker compose build
docker compose down
docker compose up -d
