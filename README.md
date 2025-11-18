# data_template_project_name

#### project_description

```
The user/developer writes a brief introduction about the project repository here.
What modules it consists of, what for, and any hints/tips that would help other
users/developers to use this package.
```

How to use this package or its modules directly, please refer to [notebooks/sample_data_template_project_name.ipynb](./notebooks/sample_data_template_project_name.ipynb)

If credentials are required in this package, please see `Required Credentials` and `Loading and Using Credentials` below.

## How to use this Template
This template is meant to assist to create a new repository project within DataLab-BMWK organization, keeping the files and folder structures as harmonious as possible in multiple project repositories. To create a project repository, all one needs is pretty much to define the name of a new repository project. The files will then be automatically updated to reflect this new name.

Here's some steps that you must do to successfully create a project repository from this template. At the end, if you're good to go and everything needed is there, feel free to delete this section. Please **READ THIS** first, before you do anything.

* First of all, **DO NOT FORK**.
* It is meant to be used from **[Use this template](https://github.com/DataLab-BMWK/data_template_project_name/generate)** feature. Click on this link. This link no longer hints at this very template once it's now new repository already.
* Choose which owner (individual or organization like `DataLab-BMWK`) and enter a name for your new project, all lowercase and underscores like this: `just_another_project`.
* Enter a description (optional). This description will also be copied into `pyproject.toml` as well as into this `README.md` (above).
* Click on `Private` Repository, unless you know what you're doing.
* Now click on `Create repository`.
* **IMPORTANT!!** Please wait for some minutes until after Github Actions is done being busy (some first CI runs). This is necessary as it'll commit new changes to this brand-new repository.
* **IMPORTANT!!** Especially if repository is not part of organization, you'll most likely notice that at least one `Github Action` workflow will fail. A special **token** will be needed to successfully finish the jobs when it just became a repository from template. Please refer to a separate documentation regarding special tokens [here](docs/configurations/github_actions_token.md).
* Ensure `main` branch is a **default** branch.
* If this new project repository requires credentials for example to access API, these must be added as secrets. See `Required Credentials` and `Loading and Using Credentials` below.
* Happy coding!


## Coverage
(TODO: coverage logo will be displayed as soon as the test coverage has stabilized - too many new developments currently.)
(Please note that the broken link to an image is here on purpose.)
<br>
![coverage](test_coverage.svg)
<br>
This test coverage includes tests needed for package `data_template_project_name`.

## Structure
* `src/data_template_project_name`: this is considered a package as `data_template_project_name`.
* `tests`: this is considered a test folder to test the contents of a package `data_template_project_name`. Please note, test folder and its subfolders must not have `__init__.py`.
* `tests/test_data_template_project_name.py`: Example calls of the very package `data_template_project_name`.
* If other modules are available, separate test files (one test file per module) are expected to be set up.

## Required Credentials
Some services require credentials in order to be able to access to their data via API, for example. If there are names and steps, please list the variable names (in other words, secrets names) here such as:
* `SOURCE_CLIENT_ID`
* `SOURCE_CLIENT_SECRET`

These are the secret/variable names that will be expected to use everywhere where necessary, including Codespaces, Actions, local servers, and Airflow DMZ repository. **Do not expose the secrets here or in any code!**


## Loading and Using Credentials
There are two approaches to store and load credentials, depending on the environment you're using:
* `DMZ Linux`: have a local `.env` file in use (`~/.env`)
* `Github Codespaces` and `Github Actions`: the credentials must be stored as `secrets`. This can be added and configured under repository's settings for `Secrets and Variables` in `github.com`.
<br><br>
For deeper examples of how credentials are used in an airflow DAG, please see BMWK Datalab Use Cases [Airflow DAGs](https://github.com/DataLab-BMWK/airflow_dmz/tree/main/airflow/dags) of `airflow_dmz` repository.

## Accessing Virtual Environment in Codespaces
For bash terminal users wanting to work in a virtual environment of data_template_project_name, enter these commands:

```
cd /workspaces/data_template_project_name
source .venv/bin/activate
```

## Accessing this Package on a Local Linux Server
This assumes you have internet connection and you already have cloned this repository. For both sub-cases below, start with these:
```
cd SOME_PATH/data_template_project_name
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

### Local Mode
```
pip install . --no-cache-dir
```

### Testing/Developing Editable Mode
```
pip install -e .[test] --no-cache-dir --force-reinstall
```

## Accessing Database in Codespaces
PostgreSQL Database is set up only inside the Codespace Development Container, which is installed via apt-get. This database in Codespaces is used for these:
* `project_db`: Project-Testing with schema `dev` and in future cases `prod`.

When container is built for the first time, and to login to the database, please click on a extension logo on the left in VS Code that looks like a database, click on "Create Connection" and enter the following info:
```
host:       127.0.0.1 (or: localhost)
Port:       5432
Username:   project_user
Password:   project_pass
Database:   project_db
```
And then, click on `+ Connect`.

Please note the following:
* Full container rebuilds do not need manual login again.
* Once Codespace is deleted - any data in this Database will be permanently deleted as well.

## Accessing Database on a Local Linux Server
This is up to the user/developer to configure PostgreSQL database. To load and use credentials to access that database, these variables must be set up in `~/.env` file or at least loaded into bash shell environment:
* `POSTGRES_USER`
* `POSTGRES_PASSWORD`
* `POSTGRES_HOSTNAME`
* `POSTGRES_PORT`
* `POSTGRES_DB_NAME`

## Working with branches (HINT: February 2025 this will be an overhaul due to newly agreed guidelines with main and other branches. Develop no longer exists.)
To those who are not aware of different branches, here's brief overview:
* `main`:       official release history, always stable and ready to deploy when needed, tagged with version number at every release. This is decided and done by at least two colleagues in charge of releases (in other words, Triin & Kent).
* `develop`:    serves as an integration branch for features. Not always stable, but gives room for corrections and further tests to get it stable again. Develop version must be updated by x.y.z+1 every time a release from main branch is made.

All other branches, always linked to tickets/issues, depending on users and developers:
* `feature 1`
* `feature 2`
* `hotfix 1`
...

If still not clear, please feel free to ask the Internet.

## Linting and Formatting
This project implements linting and formatting tools via a pre-commit hook. Most linting/formatting functionalities are covered by [Ruff](https://docs.astral.sh/ruff/). `Ruff` and other linting/formatting tools are installed in the virtual environment of `data_template_project_name`.

### Executing Linting Sources:
Executing as pre-commit hook:
```
pre-commit run --all-files
```
**Please note** that pre-commit hooks are only applied to staged changes.
### Executing Ruff
In order to check for linting and formatting errors in a bash terminal with automatic fixes, please use the commands below. This check also applies to unstaged changes as well as existing files, unless they were explicitly excluded from ruff checks.
```
ruff check . --fix
ruff format
```
Checking for included/excluded files and directories:
```
ruff check . -v
```

### Ruff Configuration:
For more details on Ruff’s configuration and rules, see our `pyproject.toml` file. In order to exclude specific rules in your repository, please check the [error code](https://docs.astral.sh/ruff/) you would like to exclude and add it in `pyproject.toml`
```
[tool.ruff.lint]
ignore = [
    "E722" #example
]
```
