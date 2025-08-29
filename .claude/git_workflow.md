# Git Workflow - Conflict Prevention Guide

## Daily Development Routine

### 1. Start of Day
```bash
# Always start with clean main branch
git checkout main
git pull origin main

# If working on feature branch, sync it regularly
git checkout your-feature-branch
git merge main  # or: git rebase main
```

### 2. Before Starting New Work
```bash
# Create new feature branch from latest main
git checkout main
git pull origin main
git checkout -b feature/your-new-feature
```

### 3. Regular Sync (Every 2-3 Days)
```bash
# Keep your feature branch up to date
git checkout main
git pull origin main
git checkout your-feature-branch
git merge main
```

### 4. Before Creating PR
```bash
# Final sync to avoid conflicts
git checkout main
git pull origin main
git checkout your-feature-branch
git merge main

# Push and create PR
git push origin your-feature-branch
```

## Conflict Prevention Rules

1. **Never work directly on main** - Always use feature branches
2. **Sync frequently** - Don't let branches diverge too much
3. **Small PRs** - Easier to review and less likely to conflict
4. **Coordinate on shared files** - Communicate when working on same areas
5. **Clean up merged branches** - Delete after successful merge

## Files That Often Cause Conflicts

- `Cargo.toml` (version numbers)
- `Cargo.lock` (dependency updates)
- `.claude/settings.local.json` (settings changes)
- Large refactoring files (when multiple people modify)

## Emergency Conflict Resolution

If conflicts occur:
1. Don't panic
2. Use `git status` to see conflicted files
3. Edit files to resolve conflicts (remove `<<<<`, `====`, `>>>>`)
4. Test that code still works
5. `git add` resolved files
6. `git commit` to complete merge