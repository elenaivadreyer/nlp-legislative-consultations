SHELL := /bin/bash

POSTGRES_VERSION := 13
PYTHON_VERSION_X_Y := $(shell python --version | cut -d " " -f 2 | cut -d "." -f 1-2)


# .PHONY: Include targets only if they do not generate a file, these targets are treated as commands to be executed
.PHONY: codespaces_only install_postgresql start_postgresql create_project_db create_venv

codespaces_only: install_postgresql create_project_db create_venv

start_codespace_services: start_postgresql

install_postgresql:
	echo "Installing PostgreSQL."
	# Commented out parts: directly from postgresql.org, cleaner approach?
	# wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add - &&  \
	# sudo sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' && \
	curl https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add - && \
    sudo echo "deb http://apt.postgresql.org/pub/repos/apt `lsb_release -cs`-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list && \
	sudo apt-get update -y && \
    sudo apt-get install -y postgresql-$(POSTGRES_VERSION) postgresql-contrib-$(POSTGRES_VERSION) postgresql-client-$(POSTGRES_VERSION) libpq-dev && \
	sudo service postgresql start && \
	echo "PostgreSQL installed and started."

create_project_db:
	# NOTE: must be a sudo user in all cases
	echo "Creating project_db in localhost PostgreSQL for project_user." && \
    sudo -u postgres psql -c "CREATE USER project_user WITH PASSWORD 'project_pass';" && \
    sudo -u postgres psql -c "CREATE DATABASE project_db WITH OWNER = project_user;" && \
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE project_db TO project_user;" && \
    sudo -u postgres psql -d project_db -c "CREATE SCHEMA IF NOT EXISTS dev;" && \
    sudo -u postgres psql -d project_db -c "GRANT ALL PRIVILEGES ON SCHEMA dev TO project_user;" && \
	echo "project_db for project_user created."

start_postgresql:
	# NOTE: You must be a sudo user to execute this command
	if pg_isready -q -h localhost -p 5432; then \
		echo "PostgreSQL is already running"; \
	else \
		echo "Starting PostgreSQL..."; \
		sudo service postgresql start; \
		sleep 20; \
		echo "PostgreSQL Started."; \
	fi

create_venv:
	if [ -d .venv ]; then rm -rf .venv; fi
	echo "Creating virtual environment now."
	python3 -m venv .venv
	source .venv/bin/activate && \
	python3 -m pip install --upgrade pip && \
	echo "Installing dependencies needed for tests." && \
	pip install -e .[test] --force-reinstall --no-cache-dir && \
	echo "Virtual environment created and dependencies installed."

run_pytest_with_cov:
	echo "Running unit tests with coverage now..."
	source .venv/bin/activate && \
	python3 -m pytest --cov=src/data_template_project_name --cov-report=term-missing --cov-report=xml --cov-append --cov-fail-under=0
	echo "Running unit tests with coverage is complete."

check_coverage_badge_for_any_changes:
	source .venv/bin/activate && \
	coverage-badge -f -o test_coverage.svg; \
	if [ $$? -eq 0 ]; then \
		echo "Badge generation successful."; \
	else \
		echo "Badge generation failed."; \
	fi; \
	if git ls-files --others --exclude-standard | grep -q '^test_coverage\.svg$$'; then \
		echo "There is newly created test_coverage.svg, please add and push to the repository. Aborting now."; \
		exit 1; \
	elif git diff --cached --name-only | grep -q '^test_coverage\.svg$$'; then \
		echo "There is a modified test_coverage.svg staged but not committed, please commit the changes. Aborting now."; \
		exit 1; \
	elif git diff --quiet --exit-code test_coverage.svg; then \
		echo "No changes to test_coverage.svg, all good."; \
	else \
		echo "There is newly a modified test_coverage.svg, please add and push to the repository. Aborting now."; \
		exit 1; \
	fi

system_information:
	echo "Information of the environment and select system."
	source .venv/bin/activate && \
	which python && \
	python --version && \
	which psql && \
	psql --version && \
	pip list && \
    printenv | sort

# Not yet used
#style:
#	source .venv/bin/activate && \
#	black .
#	flake8
#	isort .
#	pyupgrade
#	pylint
