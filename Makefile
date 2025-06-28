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
PIP := $(shell command -v pip3 || command -v pip)
VENV := venv
VENV_ACTIVATE := . $(VENV)/bin/activate

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

.PHONY: venv
venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created! Activate with: source $(VENV)/bin/activate$(NC)"

.PHONY: install
install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
ifeq ($(HAS_POETRY),yes)
	@poetry install
else
	@$(VENV_ACTIVATE) && $(PIP) install -r requirements.txt
	@[ -f requirements-dev.txt ] && $(VENV_ACTIVATE) && $(PIP) install -r requirements-dev.txt || true
endif
	@echo "$(GREEN)Dependencies installed!$(NC)"

.PHONY: install-dev
install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
ifeq ($(HAS_POETRY),yes)
	@poetry install --with dev
else
	@$(VENV_ACTIVATE) && $(PIP) install -r requirements-dev.txt || echo "No requirements-dev.txt found"
endif

.PHONY: freeze
freeze: ## Freeze dependencies
	@echo "$(BLUE)Freezing dependencies...$(NC)"
ifeq ($(HAS_POETRY),yes)
	@poetry export -f requirements.txt --output requirements.txt
else
	@$(VENV_ACTIVATE) && $(PIP) freeze > requirements.txt
endif

# ========== Development ==========

.PHONY: dev
dev: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(VENV_ACTIVATE) && python manage.py runserver
else ifeq ($(IS_FLASK),)
	@$(VENV_ACTIVATE) && flask run --debug
else ifeq ($(IS_FASTAPI),)
	@$(VENV_ACTIVATE) && uvicorn main:app --reload || uvicorn app.main:app --reload
else
	@$(VENV_ACTIVATE) && python main.py || python app.py || echo "No main entry point found"
endif

.PHONY: shell
shell: ## Start Python shell
	@echo "$(BLUE)Starting Python shell...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(VENV_ACTIVATE) && python manage.py shell
else
	@$(VENV_ACTIVATE) && ipython || python
endif

.PHONY: run
run: ## Run the application
	@echo "$(BLUE)Running application...$(NC)"
	@$(VENV_ACTIVATE) && python main.py || python app.py || python -m app

# ========== Testing ==========

.PHONY: test
test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
ifeq ($(HAS_PYTEST),yes)
	@$(VENV_ACTIVATE) && pytest
else ifeq ($(IS_DJANGO),yes)
	@$(VENV_ACTIVATE) && python manage.py test
else
	@$(VENV_ACTIVATE) && python -m unittest discover
endif

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@$(VENV_ACTIVATE) && pytest-watch || ptw || echo "Install pytest-watch for watch mode"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(VENV_ACTIVATE) && pytest --cov=. --cov-report=html --cov-report=term || python -m coverage run -m pytest

.PHONY: coverage-report
coverage-report: ## Show coverage report
	@$(VENV_ACTIVATE) && coverage report
	@echo "$(GREEN)HTML report available at htmlcov/index.html$(NC)"

# ========== Code Quality ==========

.PHONY: lint
lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	@$(VENV_ACTIVATE) && flake8 . || echo "Flake8 not installed"
	@$(VENV_ACTIVATE) && pylint **/*.py || echo "Pylint not installed"
	@$(VENV_ACTIVATE) && mypy . || echo "Mypy not installed"

.PHONY: format
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(VENV_ACTIVATE) && black . || echo "Black not installed"
	@$(VENV_ACTIVATE) && isort . || echo "isort not installed"

.PHONY: format-check
format-check: ## Check code formatting
	@echo "$(BLUE)Checking code format...$(NC)"
	@$(VENV_ACTIVATE) && black --check . || echo "Black not installed"
	@$(VENV_ACTIVATE) && isort --check-only . || echo "isort not installed"

.PHONY: type-check
type-check: ## Run type checking
	@echo "$(BLUE)Type checking...$(NC)"
	@$(VENV_ACTIVATE) && mypy . || echo "Mypy not installed"

.PHONY: security
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@$(VENV_ACTIVATE) && bandit -r . || echo "Bandit not installed"
	@$(VENV_ACTIVATE) && safety check || echo "Safety not installed"

# ========== Database (Django/SQLAlchemy) ==========

.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running migrations...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(VENV_ACTIVATE) && python manage.py migrate
else ifeq ($(HAS_ALEMBIC),yes)
	@$(VENV_ACTIVATE) && alembic upgrade head
else
	@echo "No migration system detected"
endif

.PHONY: db-makemigrations
db-makemigrations: ## Create new migrations
	@echo "$(BLUE)Creating migrations...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(VENV_ACTIVATE) && python manage.py makemigrations
else ifeq ($(HAS_ALEMBIC),yes)
	@$(VENV_ACTIVATE) && alembic revision --autogenerate -m "$(MSG)"
else
	@echo "No migration system detected"
endif

.PHONY: db-reset
db-reset: ## Reset database
	@echo "$(RED)Resetting database...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(VENV_ACTIVATE) && python manage.py flush --no-input
	@$(VENV_ACTIVATE) && python manage.py migrate
else
	@echo "Database reset not configured"
endif

.PHONY: db-seed
db-seed: ## Seed database with sample data
	@echo "$(BLUE)Seeding database...$(NC)"
ifeq ($(IS_DJANGO),yes)
	@$(VENV_ACTIVATE) && python manage.py loaddata fixtures/*.json || echo "No fixtures found"
else
	@$(VENV_ACTIVATE) && python seed.py || python -m app.seed || echo "No seed script found"
endif

# ========== API Documentation ==========

.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@$(VENV_ACTIVATE) && sphinx-build -b html docs docs/_build || echo "Sphinx not configured"

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
	@$(VENV_ACTIVATE) && flask-apispec || echo "Install flask-apispec for API docs"
else
	@echo "API documentation not configured"
endif

# ========== Django Specific ==========

ifeq ($(IS_DJANGO),yes)
.PHONY: django-admin
django-admin: ## Django admin commands (usage: make django-admin CMD="...")
	@$(VENV_ACTIVATE) && python manage.py $(CMD)

.PHONY: django-shell
django-shell: ## Django shell
	@$(VENV_ACTIVATE) && python manage.py shell_plus || python manage.py shell

.PHONY: django-static
django-static: ## Collect static files
	@$(VENV_ACTIVATE) && python manage.py collectstatic --no-input

.PHONY: django-check
django-check: ## Django system check
	@$(VENV_ACTIVATE) && python manage.py check

.PHONY: django-superuser
django-superuser: ## Create superuser
	@$(VENV_ACTIVATE) && python manage.py createsuperuser
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
	@$(VENV_ACTIVATE) && pip list --outdated
endif

.PHONY: deps-update
deps-update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
ifeq ($(HAS_POETRY),yes)
	@poetry update
else
	@$(VENV_ACTIVATE) && pip install --upgrade -r requirements.txt
endif

.PHONY: deps-audit
deps-audit: ## Audit dependencies for security issues
	@echo "$(BLUE)Auditing dependencies...$(NC)"
	@$(VENV_ACTIVATE) && pip-audit || safety check || echo "Install pip-audit or safety"

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
	@$(VENV_ACTIVATE) && python manage.py show_urls || echo "Install django-extensions"
else ifeq ($(IS_FLASK),)
	@$(VENV_ACTIVATE) && flask routes
else ifeq ($(IS_FASTAPI),)
	@echo "FastAPI routes available at: http://localhost:8000/docs"
endif

.PHONY: profile
profile: ## Profile application
	@echo "$(BLUE)Profiling application...$(NC)"
	@$(VENV_ACTIVATE) && python -m cProfile -o profile.out main.py
	@$(VENV_ACTIVATE) && python -m pstats profile.out

# ========== Git Hooks ==========

.PHONY: install-hooks
install-hooks: ## Install pre-commit hooks
	@echo "$(BLUE)Installing git hooks...$(NC)"
	@$(VENV_ACTIVATE) && pre-commit install || echo "Install pre-commit first"

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