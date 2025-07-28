# ProPostureFitness Makefile
# Convenient commands for development and deployment

.PHONY: help install test run clean build verify setup

# Default target
help:
	@echo "ProPostureFitness v5.0 - Development Commands"
	@echo "============================================"
	@echo "make setup      - Initial setup (venv + dependencies)"
	@echo "make install    - Install package in development mode"
	@echo "make test       - Run all tests"
	@echo "make run        - Run the application"
	@echo "make verify     - Verify binary functionality"
	@echo "make clean      - Clean temporary files"
	@echo "make build      - Build distribution packages"
	@echo "make format     - Format code with black"
	@echo "make lint       - Run code linting"

# Setup virtual environment and install dependencies
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Setup complete. Activate with: source venv/bin/activate"

# Install package in development mode
install:
	pip install -e .
	@echo "✅ Package installed in development mode"

# Run tests
test:
	python -m pytest tests/ -v
	@echo "✅ Tests completed"

# Run tests with coverage
test-coverage:
	python -m pytest tests/ --cov=. --cov-report=html
	@echo "✅ Coverage report generated in htmlcov/"

# Run the application
run:
	./ProPostureFitness.sh

# Run with Python directly
run-python:
	python final_working_app.py

# Verify binary
verify:
	python verify_binary.py

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.tmp" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleaned temporary files"

# Build distribution packages
build: clean
	python setup.py sdist bdist_wheel
	@echo "✅ Distribution packages built in dist/"

# Format code
format:
	black . --exclude venv
	@echo "✅ Code formatted"

# Lint code
lint:
	flake8 . --exclude venv,build,dist --max-line-length=100
	@echo "✅ Linting complete"

# Run type checking
typecheck:
	mypy . --ignore-missing-imports --exclude venv
	@echo "✅ Type checking complete"

# Full check (format, lint, typecheck, test)
check: format lint typecheck test
	@echo "✅ All checks passed!"

# Create release
release: check build
	@echo "✅ Ready for release! Check dist/ folder"
