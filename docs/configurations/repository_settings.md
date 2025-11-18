# A Guideline to Recommended Repository Settings

## Template and Repositories Creation
For reference, when a repository is created from a template, all of the repository settings will be automatically configured. This includes configuration of general settings as well as importing and activating rulesets. This page is for information, it however must be referred to scripts.

Valid especially for **template repository**: Any changes to scripts and/or documentation must be updated accordingly to stay in sync with each other.

<br><br>
For an overview, on a repository **web page** of `data_template_project_name`, go to:
```
Settings
```

For an overview, in **Github Actions** - workflows, scripts, rulesets - of `data_template_project_name`, go to:
```
data_template_project_name/.github/
```

## General

On a repository **web page** of `data_template_project_name`, go to:
```
Settings --> General
```

**Github Actions Workflow** regarding `General` Settings of a repository, it is found under:
```
data_template_project_name/.github/workflows/new_repo_configure_settings.yml
```

<br>

| Category                  | Value                     |
|---------------------------|---------------------------|
| Default Branch            | `main`                    |
| Features                  | ❌ `Wikis`<br>✅ `Issues`<br>❌ `Allow forking`<br>❌ `Sponsorships`<br>❌ `Discussions`<br>✅ `Projects`|
| Pull Requests             | ✅ `Allow merge commits`<br> ❌ `Allow squash merging`<br> ✅ `Allow rebase merging`<br> ✅ `Always suggest updating pull request branches`<br> ❌ `Allow auto-merge`<br> ✅ `Automatically delete head branches` |

## Rules: Rulesets
On a repository **web page** of `data_template_project_name`, go to:
```
Settings --> Rules --> Rulesets
```
Setup of all rulesets in **JSON** format can be found under:
```
data_template_project_name/.github/rulesets/
```
**Github Actions Workflow** regarding import and activation of rulesets, it is found under:
```
data_template_project_name/.github/workflows/new_repo_configure_settings.yml
```

For now (March 13th, 2025), one ruleset has been set up. There will be some more rulesets coming soon, especially regarding the `tags`.

### Ruleset: `default_or_main_branch_protected`
| Category                  | Value                     |
|---------------------------|---------------------------|
| Enforcement Status        | ✅ `Active`               |
| Bypass list               | `Maintain Role`: `Allow for pull requests only` |
| Target branches           | ✅ `Default`<br> ✅ `main`    |
| Branch rules              | ❌ `Restrict creations` <br> ❌ `Restrict updates`<br> ✅ `Restrict deletions` <br> ❌ `Require linear history` <br> ❌ `Require deployments to succeed`<br> ❌ `Require signed commits`<br> ✅ `Require a pull request before merging`<br> ✅ `Require status checks to pass`<br> ✅ `Block force pushes`<br> ❌ `Require code scanning results` |


| Additional settings: <br>`Require a pull request before merging` | Value                     |
|---------------------------|---------------------------|
| `Required approvals`      | ✅ `1`                    |
| `Dismiss stale pull request approvals when new commits are pushed` | ✅ |
| `Require review from Code Owners` | ❌                |
| `Require approval of the most recent reviewable push` | ❌ |
| `Require conversation resolution before merging`      | ❌ |
| `Allowed, merge methods`                              | ✅ `Merge`<br> ❌ `Squash`<br>✅ `Rebase`|

| Additional settings: <br>`Require status checks to pass` | Value                     |
|---------------------------|---------------------------|
| `Require branches to be up to date before merging`    | ✅ |
| `Do not require status checks on creation`            | ✅ |
| `Status checks that are required`                     | * `check-required-commit-keywords` <br>* `pass-required-placeholder-check` <br>(All `Github Actions`) |

* Note about `check-required-commit-keywords`:
<br>** This job can be found under github workflow named `pr_to_main_validate_commit_msgs.yml`
<br>** For more information, please see Validation of Commit Messages readme file. (link coming soon)
* Note about `pass-required-placeholder-check`:
<br>** It's a name of a job found under `.github/workflows/required_placeholder_check.yml`.
<br>** This is a placeholder for all other future checks
<br>** It simply does: ✅ `Require status checks to pass`.
<br>** This is also a placeholder for all future workflows.