.PHONY: help setup setup-env setup-docker run run-dashboard clean validate teardown

# Color output for better readability
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)===== Cosim Gym - Quick Setup =====\n$(NC)"
	@echo "$(GREEN)Available commands:$(NC)"
	@echo "  $(BLUE)make setup$(NC)              - Complete setup (env + docker)"
	@echo "  $(BLUE)make setup-env$(NC)          - Setup Python conda environment only"
	@echo "  $(BLUE)make setup-docker$(NC)       - Start Docker containers only"
	@echo "  $(BLUE)make run$(NC)                - Run simulation (after setup)"
	@echo "  $(BLUE)make run-dashboard$(NC)      - Launch Streamlit dashboard"
	@echo "  $(BLUE)make validate$(NC)           - Validate setup"
	@echo "  $(BLUE)make clean$(NC)              - Stop containers and clean"
	@echo "  $(BLUE)make teardown$(NC)           - Full cleanup (remove env+containers)"
	@echo "\n$(YELLOW)Quick Start:$(NC)"
	@echo "  make setup && make run"

## === SETUP TARGETS ===

setup: validate-tools setup-env setup-docker validate
	@echo "\n$(GREEN)✓ Setup complete! Run '$(BLUE)make run$(GREEN)' to start simulation.$(NC)"

setup-env:
	@echo "$(BLUE)Setting up Python environment...$(NC)"
	@command -v conda >/dev/null 2>&1 || { echo "$(RED)✗ Conda not found. Install Miniconda/Anaconda first.$(NC)"; exit 1; }
	@echo "Creating conda environment from environment.yml..."
	conda env create -f environment.yml --yes
	@echo "$(GREEN)✓ Python environment ready: cosim_gym$(NC)"
	@echo "$(YELLOW)  Activate with: $(BLUE)conda activate cosim_gym$(NC)"

setup-docker:
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)✗ Docker not found. Install Docker Desktop first.$(NC)"; exit 1; }
	docker compose -f src/docker-compose.yaml up -d
	@echo "$(BLUE)Waiting for Redis to be healthy...$(NC)"
	@sleep 3
	docker compose -f src/docker-compose.yaml exec -T redis redis-cli ping > /dev/null 2>&1 && \
		echo "$(GREEN)✓ Redis is running and healthy$(NC)" || \
		echo "$(YELLOW)⚠ Redis may take a few moments to start$(NC)"

## === RUN TARGETS ===

run: validate
	@echo "$(BLUE)Running simulation...$(NC)"
	conda run -n cosim_gym python src/test_script.py

run-dashboard: validate
	@echo "$(BLUE)Launching Streamlit dashboard at http://localhost:8501$(NC)"
	conda run -n cosim_gym streamlit run src/dashboard/streamlit_dashboard.py

## === VALIDATION & MONITORING ===

validate: validate-tools validate-env validate-docker
	@echo "$(GREEN)✓ All components validated$(NC)"

validate-tools:
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@command -v conda >/dev/null 2>&1 && echo "$(GREEN)✓ Conda$(NC)" || echo "$(RED)✗ Conda$(NC)"
	@command -v docker >/dev/null 2>&1 && echo "$(GREEN)✓ Docker$(NC)" || echo "$(RED)✗ Docker$(NC)"
	@command -v python >/dev/null 2>&1 && echo "$(GREEN)✓ Python$(NC)" || echo "$(RED)✗ Python$(NC)"

validate-env:
	@echo "$(BLUE)Checking Python environment...$(NC)"
	@conda env list | grep cosim_gym > /dev/null 2>&1 && \
		echo "$(GREEN)✓ Environment 'cosim_gym' exists$(NC)" || \
		echo "$(YELLOW)⚠ Environment 'cosim_gym' not found (run 'make setup-env')$(NC)"

validate-docker:
	@echo "$(BLUE)Checking Docker services...$(NC)"
	@docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null | grep -q cosim_redis && \
		echo "$(GREEN)✓ Redis is running$(NC)" || \
		echo "$(YELLOW)⚠ Redis not running (run 'make setup-docker')$(NC)"

status:
	@echo "$(BLUE)=== Current Status ===\n$(NC)"
	@echo "$(BLUE)Python Environment:$(NC)"
	@conda env list | grep cosim_gym || echo "Not created"
	@echo "\n$(BLUE)Docker Containers:$(NC)"
	@docker compose -f src/docker-compose.yaml ps || echo "Docker not running"

## === CLEANUP ===

clean:
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	docker compose -f src/docker-compose.yaml down || true
	@echo "$(GREEN)✓ Containers stopped$(NC)"

teardown: clean
	@echo "$(YELLOW)Removing Python environment...$(NC)"
	conda env remove -n cosim_gym --yes || true
	@echo "$(BLUE)Removing Docker volumes...$(NC)"
	docker volume rm cosim_gym_redis_data 2>/dev/null || true
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

logs:
	docker compose -f src/docker-compose.yaml logs -f redis

stop-redis:
	docker compose -f src/docker-compose.yaml stop redis

start-redis:
	docker compose -f src/docker-compose.yaml start redis
