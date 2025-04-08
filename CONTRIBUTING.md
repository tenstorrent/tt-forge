# Contributing Guidelines for TT-Forge

Thank you for your interest in the [TT-Forge](https://github.com/tenstorrent/tt-forge) project, we appreciate your support. TT-Forge is Tenstorrent's MLIR-based compiler. It integrates into various compiler technologies from AI/ML frameworks, to both enable running models and create custom kernel generation. The TT-Forge repository is the central hub for the various sub-projects that create the TT-Forge product. Sub-project repositories include: 

* [tt-mlir](https://github.com/tenstorrent/tt-mlir)
* [tt-xla](https://github.com/tenstorrent/tt-xla)
* [tt-npe](https://github.com/tenstorrent/tt-npe)
* [tt-thomas](https://github.com/tenstorrent/tt-thomas)
* [tt-torch](https://github.com/tenstorrent/tt-torch)
* [tt-forge-fe](https://github.com/tenstorrent/tt-forge-fe)


This document covers how to contribute to TT-Forge repositories. 

If you need to file a bug, ask for support, or make a feature request, please use the appropriate issue template:
* File a Bug
* Support 
* Request a Feature

If you are ready to make a contribution, each repository follows this process:

1. Fork the repository.
2. Clone the repository.
3. Set up the environment and build the project.
4. Make changes using the style guidelines for the repository. 
  * [Coding Guidelines](#coding-guidelines)
  * [Guidelines for Writing Effective Error Messages](#guidelines-for-writing-effective-error-messages)
  * [File Structure and Format for Legal](#file-structure-and-format-for-legal)
  * [Git, Branch Naming, and Pull Request Guidelines](#git-branch-naming-and-pull-request-guidelines)
  * [Including Documentation](#including-documentation)
5. Commit your changes.
  * Commit Changes
    * [Pre-commit](#pre-commit)
    * [Post-commit](#post-commit)
    * [CI/CD Principles](#cicd-principles)
6. Create a Pull Request 
  * [Pull Request Notes](#pull-request-notes)

## Coding Guidelines

Coding guidelines differ slightly by repo. Review the guidelines that are appropriate for the sub-project you're working with:

* [tt-mlir coding-guidelines.md](https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/coding-guidelines.md)
* tt-xla
* tt-npe
* tt-thomas
* tt-torch
* tt-forge-fe

## Guidelines for Writing Effective Error Messages
Clear and informative error messages are crucial for debugging and maintenance. A well-crafted error message can save hours of troubleshooting and make our codebase more user-friendly, especially for those less familiar with the system.

A well-written error message provides the following information to the user:
* What happened and why?
* What is the end result for the user?
* What can the user do to prevent it from happening again?

### Key Principles
#### Be Specific
Always include the actual values and conditions that caused the error. This helps to immediately identify the issue without needing to dig into the code. Vague messages like "An error occurred" or "Invalid input" don’t help in identifying the root cause.

Instead of:
```cpp
TT_FATAL(input_shape.rank() == 3, "Invalid input tensor dimensions.");
```
Write:
```cpp
TT_FATAL(input_shape.rank() == 3, "Invalid input tensor: expected 3 dimensions, but found {}.", input_shape.rank());
```
#### Explain the Issue
Provide a brief explanation of why the error occurred or why the condition is important. This helps users understand the context of the error.

Instead of:
```cpp
TT_FATAL(!input_tensor_kv.has_value(), "KV tensor cannot be passed in when sharded.");
```
Write:
```cpp
TT_FATAL(!input_tensor_kv.has_value(), "Invalid operation: KV tensor should not be provided when the input tensor is sharded. Please ensure that the KV tensor is only used in non-sharded configurations.");
```

#### Include Relevant Information
Always include relevant variable values and context in the error message to aid in quick identification of the issue. Omitting variable values or relevant details makes debugging harder.

Instead of:
```cpp
TT_FATAL(ptr != nullptr, "Pointer must not be null.");
```
Write:
```cpp
TT_FATAL(ptr != nullptr, "Failed to allocate memory: pointer is null.");
```
#### Make the Message Actionable
Ensure the error message provides clear guidance on what needs to be done to resolve the issue.
Stating what went wrong without providing guidance on how to fix it can be frustrating for users.

Instead of:
```cpp
TT_FATAL(head_size % TILE_WIDTH != 0, "Head size is invalid.");
```
Write:
```cpp
TT_FATAL(head_size % TILE_WIDTH != 0, "Invalid head size: {}. The head size must be a multiple of tile width ({}). Please adjust the dimensions accordingly.", head_size, TILE_WIDTH);
```

#### Good Example
This message clearly states the problem, includes the actual value of head_size, and offers guidance on how to fix it.
```cpp
TT_FATAL(head_size % TILE_WIDTH == 0,
         "Invalid head size: {}. The head size must be a multiple of the tile width ({}). Please adjust the dimensions accordingly.",
         head_size, TILE_WIDTH);
```

#### Style Recommendations
* Use simple, complete sentences.
* Use present tense** to describe current issues, and past tense for things that happened already.
* Use active voice** when possible; passive voice is okay for describing errors.
* Avoid using ALL CAPS and exclamation points.
* Clarify terms by adding descriptors before them. For example,<br>instead of `Specify Axis when Merge is set to No`, say `Specify the Axis parameter when the Merge option is set to No.`
* Don’t use the word "bad."** Be specific about what’s wrong. Instead of "Bad size," explain what size is needed.
* Avoid the word "please." It can make required actions sound optional.
* Start your message with the most important words that relate to the issue.

## File Structure and Format for Legal 

Every source file must have the appropriate Software Package Data Exchange (SPDX) header at the top. 

C++ header files follow the [Linux conventions](https://elixir.bootlin.com/linux/v6.5.1/source/Documentation/process/license-rules.rst#L71) for C++ source files, RST files, ASM files, and scripts. C++ header files should be treated as C++ source files and use this convention: 


```
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
```

Python files should use this convention: 


```
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
```


## Git, Branch Naming, and Pull Request Guidelines 


* Filing an issue is encouraged for any item that needs alignment or long term tracking.

* Link your issue under the `Ticket` headline in your PR description.

* Use descriptive commit messages.

* Merge commits are not allowed in our main branch. We enforce a linear
  history.

* You can use either of the following methods to merge your branch on the
  GitHub UI:
  * Squash and merge
  * Rebase and merge

### Creating a Branch

Include the user, the issue number, and optionally a description of the change.
/ and - are used as separators between user and issue number. And - and _
between issue number and description. E.g.

```
git checkout -b user-123
git checkout -b user/123
git checkout -b user-123_rename_method_x
git checkout -b user/123-add-x-unit-test
```

### Saving Your Changes

Edit the files that you want, making sure relevant unit tests pass. Then add
them in. E.g.

```
git add abc.py
git add "*.py"
```

Please avoid using `git add -A`, which is fairly error prone.

You can restore files if you need to get the original. E.g.

```
git restore abc.py
git restore '*'
git restore --staged abc.py # if the file was already added
```

```
git commit -m "Rename method x"
```

> [!NOTE] each commit on the main branch and any feature branch where multiple
> engineers collaborate should work. That is, everything compiles properly on the
> architecture used by your machine, you can run relevant code on the card, and
> relevant unit tests pass. Furthermore, for the main branch, you should run
> CI pipelines and make sure that the commit doesn't break anything important.

> You can use git log to see the sequence of commits in the branch. That allows
> you to see where your branch is relative to main, and can help you figure out
> how the commits are structured, before and after commits and rebases.

### Saving the Commit to Origin and Creating a Pull Request

You will need to push the change to origin. The command will provide a url that
you should use to create pull request. This should be done the first time you
push a change. After that you may need to set upstream to be able to push
changes in the future. E.g.

```
git push origin user-123:user-123
git branch --set-upstream-to=origin/user-123 user-123
```

or

```
git push -u branch_name
```

> [!NOTE] You may be able to push and set the upstream at the same time, but that
> assumes that you haven't rebased, which is probably not the case. The command
> would be something like:
> ```
> git push origin --set-upstream origin user-123
> ```

> If that doesn't work, you should use `branch --set-upstream-to`.

Once you have a pull request, in the UI you can run actions against the branch.
Go to Actions (https://github.com/tenstorrent/tt-metal/actions) and run the
workflows that you want against your branch. At the very least, you should run
All post-commit tests.

You can make more changes, commit them, and then if everything is set up and you
don't need to rebase, then you can just do

```
git push
```

Occasionally, and for the final set of tests before the final commit, you should
rebase your branch.

### Rebasing Your Branch

Your branch needs to be kept up to date with main via rebase. You should rebase
your local branch, and then once everything looks good, push the change. You
should not rebase your origin branch. That way, if anything goes wrong, you can
use origin to restore your branch to a good state.

> [!NOTE] For very small changes where you don't expect to create a second commit
> it might be okay to use the UI to rebase origin. However, in general, it's
> better to avoid that.

You should first make sure main is up to date:

```
git checkout main
git fetch origin
git submodule sync
git pull --rebase --prune
git submodule update --init --recursive
```

Then you can:

```
git checkout user-123
git rebase main
```

This will apply one commit at a time. Each commit is in order. If your branch
has two commits, then the first one is applied, then the second one is applied.

If there are no conflicts, everything will complete successfully. Then you can
push the changes to origin. This is done through a forced push to save the
rebase information:

```
git push -f
```

If there is any conflict with the commits being processed, you will need to
edit the files to fix the problem. Information should be printed about what to
do. It's probably a good idea not to skip commits.

Don't be surprised if changes from a subsequent commit are not there in the
first commit. For example, if you are editing the files to fix up the first of
two commits, the files will not have the edits of the second commit. When
editing files, only fix up the conflicts listed. Do not change anything else.

If you do change anything else, then `git rebase --continue` will complain and
you will probably have to restart.

Look for HEAD. The conflict will look something like:

```
<<<< HEAD
Some other edits
====
Your edits
>>>> Your branch
```

Update the file to have a single piece of working code and remove the commit
info. Make sure everything compiles and all the tests pass. Then you can
continue with the rebase.

```
git rebase --continue
```

If something is wrong enough that you want to abort the rebase and undo all the
changes, then you can start over. Do

```
git rebase --abort # go to before the rebase
```

If your local is in a bad state, you may also want to start from scratch on your
local by pulling from origin, reflogging, or checking out a specific commit via
its hash:

```
git pull --rebase # will undo changes
git reflog
git checkout <hash>
```

If none of those work you can also try:

```
git reset --hard origin/<BRANCH>
```

> [!NOTE] If you are getting errors you may need to update the origin info via
> `git branch --set-upstream-to`.

If everything goes well with all the updates. Then you can update origin:

```
git push -f
```

> [!NOTE] It's okay to have a few commits, as long as each one works on its own.
If you do want to combine commits you would want to run something like:

```
git rebase -i HEAD~10 # the number indicates how many commits you want to look at - here 10 commits
```

The latest one is at the bottom. You can use fixup or squash. Usually you want
to use fixup (indicated by f) since that discards the message. Then edit the
messages appropriately.

However, new squash and merge functionality in the UI is much easier than
doing this process manually. Therefore you should use the UI whenever possible.

### Merging to Main

You will probably need to iterate several times in terms of pushing changes and
rebasing your branch.

Once you have all of your changes working locally, your pull request (PR)
approved, and all the workflows that you want passing after a final rebase, It
is time to merge in your branch into main. This should be done in the Github UI.

Go to your PR and press the `Squash and merge` button. That will automatically
squash all of your commits, which is very useful. The button has an alternate
option to merge without squashing. You should use `Squash and merge` unless you
have a good reason not to.

After that, the UI will usually delete your branch.

## Including Documentation

You should include documentation if you are:
* Making a significant change that requires explanation for how to work with your change.

* Adding a new feature. 

## Commit Changes
This section goes over how to properly commit your contribution. 


### Pre-commit

As part of maintaining consistent codeformatting across the project, we integrated the [pre-commit](https://pre-commit.com/) framework into our workflow. Pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. It helps catch common issues early by running a set of hooks before code is committed, automating tasks like:

* Formatting code (for example, fixing trailing whitespace, enforcing end-of-file newlines)
* Running linters (for example, `clang-format`, `black`, `flake8`)
* Checking for merge conflicts or other common issues. 

For more details pre-commit, you can visit the [official documentation](https://pre-commit.com/).

For details about setting up pre-commit, refer to the pre-commit documentation for your repository: 

* tt-mlir
* tt-xla
* tt-npe
* tt-thomas
* tt-torch
* tt-forge-fe

### Post-commit

All developers are responsible for ensuring that post-commit regressions pass upon any submission to the project. Failure to ensure these tests pass will constitute a major regression and will likely mean reverting your commits.

For details on how post-commit is handled, refer to the post-commit documentation for your repository:

* tt-mlir
* tt-xla
* tt-npe
* tt-thomas
* tt-torch
* tt-forge-fe

### CI/CD Principles

Revert commits on main which fail post-commit tests immediately.
  * The names listed in the commit, and technical leads if their names are convenient and clear to find, will be pinged in their associated pipelines.
  * We will usually give a grace period during working hours depending on the load of the teams to see if the author(s) can merge a fix quickly. Otherwise, the revert will be immediate to prevent the issue from spreading to other peoples' pipelines.

There shall be a periodic discussion among the technical leads of this project concerning:
  * Certain codeowners and project-specific members review current tests in post-commit.
  * Certain codeowners and project-specific members decide whether to remove/add any current tests in post-commit as project priorities change on an ongoing basis.
  * Certain codeowners and project-specific members decide if we need to change owners or add more as project priorities change on an ongoing basis.
  * Communication channels for these decisions and meetings shall be kept internal to Tenstorrent with the intent of having such discussions in the open later.

Non-post-commit pipelines will not necessarily mean we have to revert the breaking commit, however any broken pipelines will be considered a priority bug fix. The responsibility of identifying, announcing status-tracking, and escalating broken non-post-commit pipelines will be the responsibility of codeowners whose tests are in the said non-post-commit pipeline.

In the case of the model performance test pipeline, there are codeowners for such tests. However, it is the collective responsibility of all developers to ensure that we do not regress this pipeline.


## Pull Request Notes 

For all your Pull Requests (PRs), Tenstorrent has an internal policy which your PR goes through after an initial review. For additional details about pull requests, see the [Saving the Commit to Origin and Creating a Pull Request](#saving-the-commit-to-origin-and-creating-a-pull-request) section.

The initial review encompasses the following:
* Reviewing the PR for CI/CD readiness, making sure that the code and PR at a high level make sense for the project.
* Once approved for CI/CD readiness, a Tenstorrent developer kicks off the CI/CD pipeline on your behalf.
* A 24 hour merge rule exists. Wait at least 24 hours after the PR was initially opened for review. This gives members of Tenstorrent teams that span the globe the opportunity to provide feedback on PRs.

In addition to the 24 hour rule, the following prerequisites for landing a PR exist:
* At least 1 reviewer signs off on the change
* Component owners must sign-off (GitHub will tell you if this hasn't been met)
* Green CI
* Wait at least 24 hours after opening the PR to give all tagged reviewers a chance to take a look, or at least comment on the issue that they need more time to review

```
> [!NOTE]
> Rebasing or further changes to the PR do not reset the 24 hour counter.

``` 