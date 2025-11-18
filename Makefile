SHELL := /bin/bash

PYTHON_VERSION_X_Y := $(shell python --version | cut -d " " -f 2 | cut -d "." -f 1-2)

# .PHONY: Include targets only if they do not generate a file, these targets are treated as commands to be executed
.PHONY: codespaces_only create_venv

codespaces_only: create_venv

create_venv:
	if [ -d .venv ]; then rm -rf .venv; fi
	echo "Creating virtual environment now."
	python3 -m venv .venv
	source .venv/bin/activate && \
	python3 -m pip install --upgrade pip && \
	echo "Installing dependencies needed for linting." && \
	pip install -e .[test] --force-reinstall --no-cache-dir && \
	echo "Virtual environment created and dependencies installed."

system_information:
	echo "Information of the environment and select system."
	source .venv/bin/activate && \
	which python && \
	python --version && \
	pip list && \
    printenv | sort
