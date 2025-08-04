# Makefile for managing the Docker environment

# Variables
DOCKER_COMPOSE_FILE := docker-compose-dev.yml
AIRFLOW_CUSTOM_IMAGE_NAME := airflow-custom:3.0.3
DASHBOARD_IMAGE_NAME := kevinesh/dashboard:0.1.0
DBCORE_API_IMAGE_NAME := kevinesh/dbcore-api:0.1.0

# Default target
.DEFAULT_GOAL := run

# Build the custom Docker image from the Dockerfile in the current directory
build:
	@echo "Building Docker image: $(AIRFLOW_CUSTOM_IMAGE_NAME)..."
	docker build -t $(AIRFLOW_CUSTOM_IMAGE_NAME) .
	docker build -t $(DASHBOARD_IMAGE_NAME) -f services/dashboard/Dockerfile services/dashboard
	docker build -t $(DBCORE_API_IMAGE_NAME) -f services/dbcore/Dockerfile services/dbcore

# Start the services in detached mode using the specified docker-compose file
run: build
	@echo "Starting services from $(DOCKER_COMPOSE_FILE)..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) up -d

# Stop and remove the services
down:
	@echo "Stopping services from $(DOCKER_COMPOSE_FILE)..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) down

# Restart the services
restart: down run

# Clean up dangling images and volumes
clean:
	@echo "Cleaning up..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) down -v --remove-orphans
	docker image prune -f

.PHONY: build run down restart clean
