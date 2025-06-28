# Makefile for Python Projects
# Supports Django, Flask, FastAPI, and general Python development

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# Python environment detection
PYTHON := $(shell command -v python3 || command -v python)
UV := $(shell command -v uv)
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python

# Framework detection
IS_DJANGO := $(shell [ -f "manage.py" ] && echo "yes")
IS_FLASK := $(shell grep -l "flask" requirements.txt pyproject.toml 2>/dev/null | head -1)
IS_FASTAPI := $(shell grep -l "fastapi" requirements.txt pyproject.toml 2>/dev/null | head -1)
HAS_POETRY := $(shell [ -f "pyproject.toml" ] && command -v poetry && echo "yes")
HAS_PYTEST := $(shell [ -d "tests" ] || [ -f "pytest.ini" ] && echo "yes")
HAS_ALEMBIC := $(shell [ -f "alembic.ini" ] && echo "yes")

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo '$(BLUE)Python Project Makefile$(NC)'
	@echo '======================'
	@echo ''
	@echo '$(GREEN)Detected Configuration:$(NC)'
	@echo '  Python: $(PYTHON)'
	@echo '  Django: $(IS_DJANGO)'
	@echo '  Flask: $(if $(IS_FLASK),yes,no)'
	@echo '  FastAPI: $(if $(IS_FASTAPI),yes,no)'
	@echo '  Poetry: $(HAS_POETRY)'
	@echo '  Pytest: $(HAS_PYTEST)'
	@echo ''
	@echo '$(YELLOW)Available Commands:$(NC)'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ========== Environment Setup ==========

.PHONY: check-uv
check-uv:
	@if [ -z "$(UV)" ]; then \
		echo "$(RED)UV is not installed. Please install it with:$(NC)"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "  or"; \
		echo "  pip install uv"; \
		exit 1; \
	fi

.PHONY: venv
venv: check-uv ## Create virtual environment with UV
	@echo "$(BLUE)Creating virtual environment with UV...$(NC)"
	@$(UV) venv $(VENV)
	@echo "$(GREEN)Virtual environment created! Activate with: source $(VENV)/bin/activate$(NC)"

.PHONY: install
install: check-uv ## Install dependencies with UV
	@echo "$(BLUE)Installing dependencies with UV...$(NC)"
	@$(UV) pip install -r requirements.txt
	@[ -f requirements-dev.txt ] && $(UV) pip install -r requirements-dev.txt || true
	@echo "$(GREEN)Dependencies installed!$(NC)"

.PHONY: install-dev
install-dev: check-uv ## Install development dependencies with UV
	@echo "$(BLUE)Installing development dependencies with UV...$(NC)"
	@$(UV) pip install -r requirements.txt
	@[ -f requirements-dev.txt ] && $(UV) pip install -r requirements-dev.txt || echo "No requirements-dev.txt found"

.PHONY: freeze
freeze: check-uv ## Freeze dependencies with UV
	@echo "$(BLUE)Freezing dependencies with UV...$(NC)"
	@$(UV) pip freeze > requirements.txt

.PHONY: setup
setup: check-uv ## Quick setup: create venv and install deps
	@echo "$(BLUE)Setting up project with UV...$(NC)"
	@$(UV) venv $(VENV)
	@$(UV) pip install -r requirements.txt
	@echo "$(GREEN)Setup complete! Activate with: source $(VENV)/bin/activate$(NC)"

# ========== Development ==========

.PHONY: dev
dev: check-uv ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(UV) run python manage.py runserver
else ifeq ($(IS_FLASK),)
	@$(UV) run flask run --debug
else ifeq ($(IS_FASTAPI),)
	@$(UV) run uvicorn main:app --reload || $(UV) run uvicorn app.main:app --reload
else
	@$(UV) run python main.py || $(UV) run python app.py || echo "No main entry point found"
endif

.PHONY: shell
shell: check-uv ## Start Python shell
	@echo "$(BLUE)Starting Python shell...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(UV) run python manage.py shell
else
	@$(UV) run ipython || $(UV) run python
endif

.PHONY: run
run: check-uv ## Run the application
	@echo "$(BLUE)Running application...$(NC)"
	@$(UV) run python main.py || $(UV) run python app.py || $(UV) run python -m app

# ========== Testing ==========

.PHONY: test
test: check-uv ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
ifeq ($(HAS_PYTEST),yes)
	@$(UV) run pytest
else ifeq ($(IS_DJANGO),yes)
	@$(UV) run python manage.py test
else
	@$(UV) run python -m unittest discover
endif

.PHONY: test-watch
test-watch: check-uv ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@$(UV) run pytest-watch || $(UV) run ptw || echo "Install pytest-watch for watch mode"

.PHONY: test-coverage
test-coverage: check-uv ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(UV) run pytest --cov=. --cov-report=html --cov-report=term || $(UV) run python -m coverage run -m pytest

.PHONY: coverage-report
coverage-report: check-uv ## Show coverage report
	@$(UV) run coverage report
	@echo "$(GREEN)HTML report available at htmlcov/index.html$(NC)"

# ========== Code Quality ==========

.PHONY: lint
lint: check-uv ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	@$(UV) run flake8 . || echo "Flake8 not installed"
	@$(UV) run pylint **/*.py || echo "Pylint not installed"
	@$(UV) run mypy . || echo "Mypy not installed"

.PHONY: format
format: check-uv ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(UV) run black . || echo "Black not installed"
	@$(UV) run isort . || echo "isort not installed"

.PHONY: format-check
format-check: check-uv ## Check code formatting
	@echo "$(BLUE)Checking code format...$(NC)"
	@$(UV) run black --check . || echo "Black not installed"
	@$(UV) run isort --check-only . || echo "isort not installed"

.PHONY: type-check
type-check: check-uv ## Run type checking
	@echo "$(BLUE)Type checking...$(NC)"
	@$(UV) run mypy . || echo "Mypy not installed"

.PHONY: security
security: check-uv ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@$(UV) run bandit -r . || echo "Bandit not installed"
	@$(UV) run safety check || echo "Safety not installed"

# ========== Database (Django/SQLAlchemy) ==========

.PHONY: db-migrate
db-migrate: check-uv ## Run database migrations
	@echo "$(BLUE)Running migrations...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(UV) run python manage.py migrate
else ifeq ($(HAS_ALEMBIC),yes)
	@$(UV) run alembic upgrade head
else
	@echo "No migration system detected"
endif

.PHONY: db-makemigrations
db-makemigrations: check-uv ## Create new migrations
	@echo "$(BLUE)Creating migrations...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(UV) run python manage.py makemigrations
else ifeq ($(HAS_ALEMBIC),yes)
	@$(UV) run alembic revision --autogenerate -m "$(MSG)"
else
	@echo "No migration system detected"
endif

.PHONY: db-reset
db-reset: check-uv ## Reset database
	@echo "$(RED)Resetting database...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(UV) run python manage.py flush --no-input
	@$(UV) run python manage.py migrate
else
	@echo "Database reset not configured"
endif

.PHONY: db-seed
db-seed: check-uv ## Seed database with sample data
	@echo "$(BLUE)Seeding database...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(UV) run python manage.py loaddata fixtures/*.json || echo "No fixtures found"
else
	@$(UV) run python seed.py || $(UV) run python -m app.seed || echo "No seed script found"
endif

# ========== API Documentation ==========

.PHONY: docs
docs: check-uv ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@$(UV) run sphinx-build -b html docs docs/_build || echo "Sphinx not configured"

.PHONY: docs-serve
docs-serve: ## Serve documentation
	@echo "$(BLUE)Serving documentation...$(NC)"
	@cd docs/_build/html && python -m http.server

.PHONY: api-docs
api-docs: ## Generate API documentation
	@echo "$(BLUE)Generating API documentation...$(NC)"
ifeq ($(IS_FASTAPI),)
	@echo "FastAPI automatic docs at: http://localhost:8000/docs"
else ifeq ($(IS_FLASK),)
	@$(UV) run flask-apispec || echo "Install flask-apispec for API docs"
else
	@echo "API documentation not configured"
endif

# ========== Django Specific ==========

ifeq ($(IS_DJANGO),yes)
.PHONY: django-admin
django-admin: ## Django admin commands (usage: make django-admin CMD="...")
	@$(UV) run python manage.py $(CMD)

.PHONY: django-shell
django-shell: ## Django shell
	@$(UV) run python manage.py shell_plus || python manage.py shell

.PHONY: django-static
django-static: ## Collect static files
	@$(UV) run python manage.py collectstatic --no-input

.PHONY: django-check
django-check: ## Django system check
	@$(UV) run python manage.py check

.PHONY: django-superuser
django-superuser: ## Create superuser
	@$(UV) run python manage.py createsuperuser
endif

# ========== Deployment ==========

.PHONY: build
build: ## Build application
	@echo "$(BLUE)Building application...$(NC)"
ifeq ($(HAS_POETRY),yes)
	@poetry build
else
	@$(PYTHON) setup.py sdist bdist_wheel || echo "No setup.py found"
endif

.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t $(shell basename $(CURDIR)):latest .

.PHONY: docker-run
docker-run: ## Run in Docker
	@echo "$(BLUE)Running Docker container...$(NC)"
	@docker run -p 8000:8000 $(shell basename $(CURDIR)):latest

# ========== Maintenance ==========

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ build/ dist/ 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

.PHONY: deps-check
deps-check: ## Check for outdated dependencies
	@echo "$(BLUE)Checking dependencies...$(NC)"
ifeq ($(HAS_POETRY),yes)
	@poetry show --outdated
else
	@$(UV) run pip list --outdated
endif

.PHONY: deps-update
deps-update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
ifeq ($(HAS_POETRY),yes)
	@poetry update
else
	@$(UV) run pip install --upgrade -r requirements.txt
endif

.PHONY: deps-audit
deps-audit: ## Audit dependencies for security issues
	@echo "$(BLUE)Auditing dependencies...$(NC)"
	@$(UV) run pip-audit || safety check || echo "Install pip-audit or safety"

# ========== Utilities ==========

.PHONY: env-check
env-check: ## Check environment variables
	@echo "$(BLUE)Environment Variables:$(NC)"
	@[ -f .env ] && echo "✓ .env exists" || echo "✗ .env missing"
	@[ -f .env.example ] && echo "✓ .env.example exists" || echo "✗ .env.example missing"

.PHONY: routes
routes: ## List API routes
	@echo "$(BLUE)API Routes:$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(UV) run python manage.py show_urls || echo "Install django-extensions"
else ifeq ($(IS_FLASK),)
	@$(UV) run flask routes
else ifeq ($(IS_FASTAPI),)
	@echo "FastAPI routes available at: http://localhost:8000/docs"
endif

.PHONY: profile
profile: ## Profile application
	@echo "$(BLUE)Profiling application...$(NC)"
	@$(UV) run python -m cProfile -o profile.out main.py
	@$(UV) run python -m pstats profile.out

# ========== Git Hooks ==========

.PHONY: install-hooks
install-hooks: ## Install pre-commit hooks
	@echo "$(BLUE)Installing git hooks...$(NC)"
	@$(UV) run pre-commit install || echo "Install pre-commit first"

.PHONY: pre-commit
pre-commit: format lint type-check test ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

# ========== Help Aliases ==========

.PHONY: i
i: install ## Alias for install

.PHONY: d
d: dev ## Alias for dev

.PHONY: t
t: test ## Alias for test

.PHONY: l
l: lint ## Alias for lint

.PHONY: f
f: format ## Alias for format