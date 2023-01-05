#!/bin/bash

echo "killing old docker processes"
docker-compose rm -fs

echo "building docker containers"
docker-compose --compatibility up --build -d