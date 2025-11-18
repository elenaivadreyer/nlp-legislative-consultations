# Workflows with destination branch `main`

## Overview
This documentation fully reflects of what is implemented in github action workflows, mostly in this file:

```
.github/workflows/pr_to_main_validate_commits_and_version.yml
```

Any changes in workflows shall have this documentation adapted.

Any branch being merged to `main`, it must always go through `Pull Request` and must have requirements met before merging is possible. All other approaches, like direct push to main, are blocked.

Exception: `Bypass` is possible, but only for `Maintainer` and only in `Pull Request`.


## All Tests

Tests are considered in

```
.github/workflows/main.yml
```

This not only is a requirement for a PR to pass, but also anytime anything is pushed to `main` branch, this `main.yml` is automatically triggered.

For more detailed information on tests, what tests are there, please refer to this `YML` file.

## Validation of Commit Messages
The validation will be checked only in the pull request with destination branch `main`. Only the commit messages between branching from main and merging to main are considered, in other words all new commits as seen in the `Pull Request`.

For the checks, please note that for every criteria, a requirement met in `ANY` commit message of all these PR commit messages is enough.

The validation checks are split into two steps (possible that more will come later):
### Semantic Versioning Cases
* It checks for any at least one required keyword needed for `Semantic Versioning` cases later. It must consist of any of these (case insensitive):
```
* MAJOR CHANGE
* BREAKING CHANGE
* MINOR CHANGE
* PATCH CHANGE
* HOTFIX
* BUG FIX (or BUGFIX)
```

### Inclusion of Issue Number

Issue Number must be mentioned in `ANY` commit message of all commits in a Pull Request. Issue number must be followed by a hashtag `#`, for example like this in a commit message:

```
#49: README is refactored for better readability
```

## Checking that the version in a branch is newer than that of main branch in PR
For time being, the updates to version are **manual**, semantic versioning is followed. Adapting the version is one of the requirements for PR to be allowed to be merged. And a version number must be a newer one.

Versions like `1.0.0`, `1.0.1.dev0`, `1.0.1`... `x.y.z`, `a.b.c.devX` etc. are fully considered and respected according to a guideline listed in [Guidelines: Version and Release Management](../guidelines/version_and_release_management.md).