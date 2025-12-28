#!/bin/bash
#
# Install the pre-commit hook for WeaverTools
#
# This script creates a symbolic link from .git/hooks/pre-commit to the
# hooks/pre-commit script in this repository.
#
# Usage:
#   ./hooks/install.sh           # Install the hook
#   ./hooks/install.sh --remove  # Remove the hook
#

set -e

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$REPO_ROOT" ]; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Get the git hooks directory
# Handle worktrees where .git is a file pointing to the actual git dir
GIT_DIR=$(git rev-parse --git-dir 2>/dev/null)
HOOKS_DIR="$GIT_DIR/hooks"

SOURCE_HOOK="$REPO_ROOT/hooks/pre-commit"
TARGET_HOOK="$HOOKS_DIR/pre-commit"

# Handle removal
if [ "$1" = "--remove" ] || [ "$1" = "-r" ]; then
    if [ -L "$TARGET_HOOK" ]; then
        rm "$TARGET_HOOK"
        echo -e "${GREEN}Pre-commit hook removed successfully.${NC}"
    elif [ -f "$TARGET_HOOK" ]; then
        echo -e "${YELLOW}Warning: $TARGET_HOOK exists but is not a symlink.${NC}"
        echo "Remove it manually if you want to uninstall."
        exit 1
    else
        echo -e "${YELLOW}Pre-commit hook is not installed.${NC}"
    fi
    exit 0
fi

# Check if source hook exists
if [ ! -f "$SOURCE_HOOK" ]; then
    echo -e "${RED}Error: Source hook not found at $SOURCE_HOOK${NC}"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Check if a pre-commit hook already exists
if [ -f "$TARGET_HOOK" ]; then
    if [ -L "$TARGET_HOOK" ]; then
        # It's a symlink - check if it points to our hook
        current_target=$(readlink -f "$TARGET_HOOK" 2>/dev/null || readlink "$TARGET_HOOK")
        if [ "$current_target" = "$(readlink -f "$SOURCE_HOOK" 2>/dev/null || echo "$SOURCE_HOOK")" ]; then
            echo -e "${GREEN}Pre-commit hook is already installed.${NC}"
            exit 0
        fi
    fi

    echo -e "${YELLOW}Warning: A pre-commit hook already exists at $TARGET_HOOK${NC}"
    read -p "Do you want to replace it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi

    # Backup existing hook
    backup_file="$TARGET_HOOK.backup.$(date +%Y%m%d%H%M%S)"
    mv "$TARGET_HOOK" "$backup_file"
    echo -e "Backed up existing hook to: ${YELLOW}$backup_file${NC}"
fi

# Create symlink
ln -s "$SOURCE_HOOK" "$TARGET_HOOK"

# Make source executable (just in case)
chmod +x "$SOURCE_HOOK"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Pre-commit hook installed successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "The hook will scan staged files for potential secrets before each commit."
echo ""
echo "To bypass the hook (NOT recommended):"
echo "  git commit --no-verify"
echo ""
echo "To remove the hook:"
echo "  ./hooks/install.sh --remove"
echo ""
