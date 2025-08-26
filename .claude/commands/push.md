# Push ³ŞóÉ

## ú,„j×Ã·åÕíü

### 1. ¯¤Ã¯×Ã·åclippy + format + push	
```bash
# YyfnfJ’îcWfÕ©üŞÃÈWf×Ã·å
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged && \
cargo fmt --all && \
git add -A && \
git commit -m "style: Apply clippy fixes and formatting" && \
git push origin $(git branch --show-current)
```

### 2. µ„×Ã·åºWjL‰	
```bash
# Step 1: Clippy Á§Ã¯hêÕîc
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged

# Step 2: Õ©üŞÃÈ
cargo fmt --all

# Step 3: 	ôº
git diff
git status

# Step 4: ³ßÃÈ
git add -A
git commit -m "style: Apply clippy fixes and formatting

- Resolve clippy suggestions
- Apply rustfmt formatting
- Ensure code quality standards

> Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Step 5: ×Ã·å
git push origin $(git branch --show-current)
```

## ×éÃÈÕ©üà%Õ£üÁãü-š

### Ubuntu/Linux
```bash
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged
cargo fmt --all
```

### macOS
```bash
cargo clippy --all-targets --features "linalg-netlib" --target x86_64-apple-darwin --fix --allow-dirty --allow-staged
cargo fmt --all
```

### Windows
```bash
cargo clippy --all-targets --no-default-features --fix --allow-dirty --allow-staged
cargo fmt --all
```

## ŒhÁ§Ã¯Œn×Ã·å

```bash
# YyfnÁ§Ã¯’ŸLWfK‰×Ã·å
cargo clippy --all-targets --features "linalg-netlib" -- -D warnings && \
cargo fmt --all --check && \
cargo test --features "linalg-netlib" && \
git add -A && \
git commit -m "chore: Code quality improvements" && \
git push origin $(git branch --show-current)
```

## ¨¤ê¢¹-šª×·çó	

`.bashrc` ~_o `.zshrc` kı :

```bash
alias rustpush='cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged && cargo fmt --all && git add -A && git commit -m "style: Apply clippy fixes and formatting" && git push origin $(git branch --show-current)'
```

(¹Õ:
```bash
rustpush
```

## ÈéÖë·åüÆ£ó°

### Clippy ¨éüLêÕîcgMjD4
```bash
# fJ’º
cargo clippy --all-targets --features "linalg-netlib"

# KÕîcŒk¦ŸL
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged
```

### Õ©üŞÃÈÁ§Ã¯n
```bash
# Á§Ã¯n	ôjW	
cargo fmt --all --check
```

### yšÕ¡¤ënæ
```bash
# yšÕ¡¤ënÕ©üŞÃÈ
cargo fmt -- src/lib.rs src/tensor/mod.rs

# yšÕ¡¤ënı Wf³ßÃÈ
git add src/lib.rs src/tensor/mod.rs
git commit -m "style: Format specific modules"
git push origin $(git branch --show-current)
```