# nlp_legislative_consultations

#### NLP data analysis project analyzing the influence of stakeholder comments on final legal drafts.

This repository provides a framework for performing NLP analysis on legislative consultation documents. The project follows a standard data analysis workflow with directories organized by purpose.

## Project Structure

```
nlp_legislative_consultations/
├── data/
│   ├── raw/              # Original data files (Excel, CSV, etc.)
│   └── processed/        # Cleaned and preprocessed data
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── scripts/              # Python scripts for the NLP pipeline
│   ├── data_acquisition
│   ├── text_preprocessing
│   ├── modeling
│   └── evaluation
├── results/              # Output files (predictions, metrics, visualizations)
└── requirements.txt      # Python dependencies
```

### Directory Guide

* **data/**: Store your data files here. Raw data goes in `raw/`, processed data in `processed/`
* **notebooks/**: Jupyter notebooks for exploratory data analysis and prototyping
* **scripts/**: Production-ready Python scripts for data processing and modeling
* **results/**: Model outputs, evaluation metrics, and visualizations

For usage examples, see [notebooks/sample_nlp_legislative_consultations.ipynb](./notebooks/sample_nlp_legislative_consultations.ipynb)

## Getting Started

### Setup on Codespaces

For bash terminal users in Codespaces:

```bash
cd /workspaces/nlp_legislative_consultations
source .venv/bin/activate
```

### Setup on Local Linux Server

1. Clone the repository and navigate to it:
```bash
cd /path/to/nlp_legislative_consultations
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or use the Makefile:
```bash
make create_venv  # Creates venv and installs dependencies
```

## Working with the Project

### Running Scripts

Execute Python scripts from the project root:

```bash
python scripts/your_script.py
```

### Using Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Then open notebooks from the `notebooks/` directory.

## Linting and Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting via pre-commit hooks.

### Setup pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

### Manual linting

```bash
ruff check . --fix
ruff format
```

Check which files are included/excluded:
```bash
ruff check . -v
```

### Configuration

Ruff configuration is in `pyproject.toml`. To exclude specific rules:

```toml
[tool.ruff.lint]
ignore = [
    "E722"  # example rule to ignore
]
```

## Dependencies

Core dependencies are listed in `requirements.txt`:
- **openpyxl**: Excel file handling
- **pandas**: Data manipulation and analysis
- **requests**: HTTP requests for web scraping
- **beautifulsoup4**: HTML/XML parsing
- **xlrd**: Legacy Excel file support

Development dependencies:
- **ruff**: Linting and formatting
- **pre-commit**: Git hooks for code quality
