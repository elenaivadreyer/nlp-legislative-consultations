# Token for GitHub Actions

## Introduction

This page assumes that you have a `GitHub Team`-Plan for the said organization, and **no** `Enterprise`-Plan is used (for any reason). <br><br>
A default built-in token (`GITHUB_TOKEN`) alone is not sufficient for certain tasks, especially when GitHub Action Workflows attempt to do exactly these tasks:
* Configuring Repository General Settings
* Importing Rulesets to Repository Settings

Therefore, an extra and separate token is needed to be created, and stored locally (for reference) and in a target repository as a **secret**. The name of such token for Github Actions is:
```
GH_ACTIONS_WORKFLOW_TOKEN
```


## How to Create a Token

| First Steps:                                  |
|-----------------------------------------------|
| Be logged in to `github.com`                  |
| Above right, click on `User` logo             |
| Click on `Settings`                           |
| You are on `Your personal account` page       |
| Go to `Developer Settings`                    |
| Go to `Personal access tokens`                |
| Go to `Fine-grained tokens`                   |
| Click on `Generate new token`                 |


### Generating Token
You are on page `New fine-grained personal access token`. Follow these steps to successfully generate a token.

| Category                  | Input                             |
|---------------------------|-----------------------------------|
| Token name                | `GH_ACTIONS_WORKFLOW_TOKEN`       |
| Resource owner            | One of these two:<br><ul><li> `DataLab-BMWK` (should be as default as possible)</li><li>`SomeUserName` (if newly created repository in user's account - mostly for testing)</li></ul> |
| Expiration                | Recommended: `90 days` or `180 days`<br> **WARNING: NEVER click on `No expiration`!** |
| Repository access         | `All repositories`<br> **NOTE**: To be discussed if we want to actually restrict to `selected repositories`.                |


<br>

| Repository Permissions    | Access                    | Hint               |
|---------------------------|---------------------------|--------------------|
| Actions                   | `Read and write`          | To trigger or modify GitHub Actions workflows.|
| Administration            | `Read and write`          | Required to create, edit, and activate repository rulesets. |
| Contents                  | `Read and write`          | To read yml files from repository, to modify files in repository.|
| Metadata                  | `Read-only`               | Always mandatory. |
| Workflows                 | `Read and write`          | Update GitHub Action workflow files.|
| **All other Permissions** | `No access`               | Not relevant for specific purposes.|

<br>

| Account Permissions   | Access            | Hint               |
|-----------------------|-------------------|--------------------|
| **All Permissions**   | `No access`       | Not relevant for specific purposes. |

### Saving Token

Once token is generated, make sure you save it somewhere safe with name of that token. (Local Notebook, for example). Github will show this newly generated token password only once.

## How to get Repository to Access to this Token
This token must be stored as **secret** for `Actions`. Where to store it as a **secret** depends on these main criteria:
* Who is the `Resource Owner` of this **token**?
* Does this token have an access to the said repository?
* Which `Settings` to use?
    * `Organization Settings`: if a said repository belong to the said organization, use this approach. This option should be used as a **default**, if possible.
    * `Repository Settings`: use this approach if a said repository does not belong to the said organization.

NOTE:
* this has already been tested with **Repository** `Settings` (with User's token on specific repository). GitHub Actions have been successful.
* **Organization** `Settings` not yet fully tested and verified.

Here's how-to:

| Step-by-Step                              | Hints                                 |
|-------------------------------------------|---------------------------------------|
| Go to `Settings`                          | `Organization Settings` <br>`Repository Settings` |
| Click on `Secrets and variables`          | |
| Go to `Actions`                           | |
| Click on `New secret`                     | `New organization secret`<br>`New repository secret` ||
| Enter **Name**: <br> `GH_ACTIONS_WORKFLOW_TOKEN` | |
| Enter **Value** (it is the token password) | |
| **Repository access**: <br>`Private repositories` | **ONLY** in  `Organization Settings` |
| Click on `Add secret`

## Re-running failed GitHub Action Workflows
If you had a failing Github Action workflow before token-related configurations, go there again and re-run the failed jobs. This is especially relevant when the repository is just created from the `Template`.