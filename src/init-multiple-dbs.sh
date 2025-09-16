#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-E0SQL
  CREATE USER mlflow WITH PASSWORD 'mlflow';
  CREATE DATABASE mlflow;
  GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
E0SQL