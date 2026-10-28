
#!/bin/bash
set -e

cd "$(dirname "$0")" || exit 1
source ../git.sh

[ -n "$GH_TOKEN" ] || { echo "no GH_TOKEN"; exit 1; }
[ -n "$GH_USER" ] || { echo "no GH_USER"; exit 1; }

REPO="github.com/underdebug/2610-SpeedingUp.git"
BRANCH="master"
URL="https://${GH_USER}:${GH_TOKEN}@${REPO}"
MSG="${1:-update $(date '+%F %H:%M')}"

# Initialize repository if needed
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    git init
    git config user.name "$GH_USER"
    git config user.email "mingjie2026@gmail.com"
fi

# Detect remote branch
REMOTE_BRANCHES=$(git ls-remote --heads "$URL")

if grep -q "refs/heads/$BRANCH$" <<< "$REMOTE_BRANCHES"; then
    echo "Remote branch: $BRANCH"
elif grep -q 'refs/heads/main$' <<< "$REMOTE_BRANCHES"; then
    BRANCH="main"
    echo "Using existing remote branch: $BRANCH"
else
    echo "Remote branch not found. First push will create $BRANCH."
fi

# Select local branch safely
CURRENT_BRANCH=$(git symbolic-ref --quiet --short HEAD || true)

if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git switch "$BRANCH"
    elif git rev-parse --verify HEAD >/dev/null 2>&1; then
        git switch -c "$BRANCH"
    else
        git symbolic-ref HEAD "refs/heads/$BRANCH"
    fi
fi

# Ignore build and Python cache directories
touch .gitignore
for pattern in 'build/' '__pycache__/'; do
    grep -Fxq "$pattern" .gitignore || echo "$pattern" >> .gitignore
done

# Remove previously tracked build/cache files from Git
# Keep local files until push succeeds
while IFS= read -r -d '' file; do
    case "/$file/" in
        */build/*|*/__pycache__/*)
            git rm --cached -q --ignore-unmatch -- "$file"
            ;;
    esac
done < <(git ls-files -z)

# Stage changes
git add -A

# Commit local changes before synchronization
if ! git diff --cached --quiet; then
    git commit -m "$MSG"
else
    echo "no changes"
fi

# Retrieve remote changes only when branch exists
if grep -q "refs/heads/$BRANCH$" <<< "$REMOTE_BRANCHES"; then
    git fetch "$URL" "$BRANCH"
    git merge FETCH_HEAD --no-edit --allow-unrelated-histories
else
    echo "Skipping fetch: remote branch does not exist."
fi

# Push changes
git push "$URL" "HEAD:refs/heads/$BRANCH"

# Clean local directories only after successful push
echo "Cleaning local build and __pycache__ directories..."

find . -type d \( -name build -o -name __pycache__ \) \
    -not -path './.git/*' \
    -prune -print -exec rm -rf -- {} +

echo "Upload complete; build and __pycache__ directories cleaned."

