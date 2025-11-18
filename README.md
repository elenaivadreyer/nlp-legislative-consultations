# nlp_legislative_consultations

#### NLP project analyzing the influence of stakeholder comments on final legal drafts.

This repository provides a structured framework for performing NLP analysis on legislative consultation documents. The project follows a standard NLP pipeline approach with distinct modules for each stage of the process.

## Project Structure

The repository is organized around the standard steps of an NLP pipeline:

* **data_acquisition**: Handling data from Excel files and web sources
* **text_preprocessing**: Text cleaning, tokenization, and normalization
* **modeling**: Text understanding, feature extraction, and model training
* **evaluation**: Model evaluation and metrics

For usage examples, please refer to [notebooks/sample_nlp_legislative_consultations.ipynb](./notebooks/sample_nlp_legislative_consultations.ipynb)

## Getting Started

### Accessing Virtual Environment in Codespaces
For bash terminal users wanting to work in a virtual environment of nlp_legislative_consultations, enter these commands:

```bash
cd /workspaces/nlp_legislative_consultations
source .venv/bin/activate
```

### Accessing this Package on a Local Linux Server
This assumes you have internet connection and you already have cloned this repository. Start with these:
```bash
cd SOME_PATH/nlp_legislative_consultations
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

#### Local Mode
```bash
pip install . --no-cache-dir
```

#### Development Mode
```bash
pip install -e .[test] --no-cache-dir --force-reinstall
```

## Structure
* `src/nlp_legislative_consultations`: The main package with NLP pipeline modules:
  * `data_acquisition/`: Data loading from Excel files and web scraping
  * `text_preprocessing/`: Text cleaning and preprocessing
  * `modeling/`: Text understanding and modeling
  * `evaluation/`: Model evaluation and metrics
* `notebooks/`: Jupyter notebooks with usage examples

## Linting and Formatting
This project implements linting and formatting tools via a pre-commit hook using [Ruff](https://docs.astral.sh/ruff/).

### Executing Linting:
Execute as pre-commit hook:
```bash
pre-commit run --all-files
```

### Executing Ruff directly:
To check for linting and formatting errors with automatic fixes:
```bash
ruff check . --fix
ruff format
```

Checking for included/excluded files and directories:
```bash
ruff check . -v
```

### Ruff Configuration:
For more details on Ruff's configuration and rules, see our `pyproject.toml` file. To exclude specific rules in your repository, check the [error code](https://docs.astral.sh/ruff/) you would like to exclude and add it in `pyproject.toml`:
```toml
[tool.ruff.lint]
ignore = [
    "E722" #example
]
```
