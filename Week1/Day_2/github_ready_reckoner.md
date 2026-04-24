# GitHub Ready Reckoner: Essential Git Commands

This guide provides a quick reference for common Git commands used with GitHub repositories.

## Setting Up a Repository

### `git init`

Initializes a new Git repository in the current directory.

```bash
git init
```

**What it does:**

- Creates a hidden `.git` directory that contains all the necessary repository metadata
- Enables Git version control for the directory
- Does not create any commits or connect to remote repositories

**Usage example:**

```bash
mkdir my-new-project
cd my-new-project
git init
```

## Tracking Changes

### `git status`

Shows the current state of your working directory and staging area.

```bash
git status
```

**What it shows:**

- Files that have been modified but not staged
- Files that are staged and ready to be committed
- Untracked files
- Branch information

**Tips:**

- Use `git status -s` for a short, simplified output
- Green files are staged, red files are modified but not staged

## Staging Changes

### `git add`

Adds changes to the staging area for the next commit.

```bash
git add filename    # Add a specific file
git add .           # Add all changes in the current directory
git add -A          # Add all changes in the repository
```

**To undo git add:**

```bash
git reset filename  # Unstage a specific file
git reset           # Unstage all changes
```

**Tips:**

- `git add -p` allows you to interactively choose portions of files to stage

I'd be happy to add information about `.gitignore` to the GitHub Ready Reckoner markdown document. Here's how I would integrate it:

## Git Ignore

The `.gitignore` file tells Git which files or directories to ignore in a project.

### Creating a .gitignore file

```
# Create a .gitignore file in your repository root
touch .gitignore
```

### Common .gitignore patterns

```
# Ignore specific files
config.json
secrets.txt

# Ignore file types
*.log
*.tmp

# Ignore directories
node_modules/
dist/
build/

# Ignore all files in a directory but keep the directory
logs/*
!logs/.gitkeep
```

### Global .gitignore

You can also set up a global .gitignore for all repositories:

```
git config --global core.excludesfile ~/.gitignore_global
```

### Templates

GitHub maintains a collection of useful .gitignore templates for different programming languages and environments at: https://github.com/github/gitignore

## Making Commits

### `git commit`

Creates a snapshot of the staged changes.

```bash
git commit -m "Your descriptive commit message"
```

**To undo a commit:**

If you haven't pushed yet:

```bash
git reset --soft HEAD~1  # Undo the commit but keep the changes staged
git reset HEAD~1         # Undo the commit and unstage the changes
git reset --hard HEAD~1  # Undo the commit and discard all changes (caution!)
```

If you've already pushed:

```bash
git revert HEAD          # Creates a new commit that undoes the last commit
```

**Tips:**

- Use `git commit --amend` to modify your most recent commit

## Viewing History

### `git log`

Shows the commit history.

```bash
git log -2              # Show only the last 2 commits
```

**Common options:**

```bash
git log --oneline       # Compact view, one line per commit
git log --graph         # Show branch and merge history
git log --stat          # Show statistics of changes
git log -p              # Show detailed changes
```

## Reverting to Previous Commits

### Revert to a previous commit

Different ways to go back to a previous state:

**Temporarily check out a previous commit:**

```bash
git checkout <commit-hash>  # Detached HEAD state
```

**Create a new branch from a previous commit:**

```bash
git checkout -b <new-branch-name> <commit-hash>
```

**Revert a commit (safest option):**

```bash
git revert <commit-hash>  # Creates a new commit that undoes the specified commit
```

**Reset to a previous commit:**

```bash
git reset --soft <commit-hash>  # Keep changes staged
git reset <commit-hash>         # Keep changes unstaged
git reset --hard <commit-hash>  # Discard all changes (dangerous!)
```

## Working with Remote Repositories

### Using Personal Access Tokens (PAT)

GitHub has phased out password authentication for Git operations. You'll need to use a Personal Access Token instead.

#### Setting up a Personal Access Token:

1. Go to GitHub → Click on your profile in top-right → Settings → Developer settings → Personal access tokens → Generate new token
2. Select appropriate scopes (at minimum, select "repo")
3. Generate the token and copy it immediately (it won't be shown again)

### Adding a remote repository

```bash
git remote set-url origin https://USERNAME:TOKEN@github.com/USERNAME/REPOSITORY.git
```

**Check existing remotes:**

```bash
git remote -v
```

### Pushing to GitHub

```bash
git push <remote-name> <branch-name>
git push origin main
```

**First push with branch tracking:**

```bash
git push -u origin main  # Sets up tracking for the branch
```

**Force push (use with caution):**

```bash
git push --force origin main  # Overwrites remote branch (dangerous!)
```

I'll introduce information about `git pull` to the GitHub Ready Reckoner markdown document. This would fit well in the "Working with Remote Repositories" section:

## Git Pull

The `git pull` command fetches changes from a remote repository and integrates them into your local branch.

### Basic pull command

```
git pull <remote-name> <branch-name>
git pull origin main
```

### What it does

- Combines `git fetch` (downloads changes) and `git merge` (integrates changes) in one command
- Updates your current branch with changes from the remote branch

### Pull options

```
git pull --rebase              # Applies your local commits on top of the remote changes
git pull --no-commit           # Fetches and merges but doesn't create an automatic commit
git pull --ff-only             # Only pulls if a fast-forward merge is possible
```

### Handling merge conflicts

If there are conflicts during a pull:

1. Git will mark the conflicted files
2. Manually resolve conflicts in each file
3. Use `git add` to mark them as resolved
4. Complete the merge with `git commit`

### Best practices

- Always commit or stash local changes before pulling
- Use `git pull --rebase` to keep a cleaner history when working on shared branches
- Consider using `git fetch` followed by `git merge` for more control over the integration process

This section provides essential information about pulling changes from remote repositories, which is a fundamental part of collaborative Git workflows.

## Common Workflows

### Starting a new project

```bash
mkdir project-name
cd project-name
git init
# Create some files
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/project-name.git
git push -u origin main
```

### Making changes to an existing project

```bash
git pull              # Get latest changes
# Make your changes
git status            # Check what's changed
git add .             # Stage changes
git commit -m "Description of changes"
git push origin main  # Share your changes
```

## Troubleshooting

### Undoing a git operation

Most Git operations can be undone, but different commands require different approaches:

- **Undo git add**: `git reset`
- **Undo last commit**: `git reset --soft HEAD~1`
- **Undo a pushed commit**: `git revert <commit-hash>`
- **Undo a git merge**: `git reset --hard ORIG_HEAD` (immediately after merge)

### When things go wrong

If you've made a mistake and aren't sure what to do:

1. Don't panic! Git rarely loses data permanently
2. Check `git status` to understand your current state
3. Use `git reflog` to see a history of all your recent actions
4. Make a backup of your entire repository (just copy the folder)
5. Consider asking for help on Stack Overflow or from a colleague
