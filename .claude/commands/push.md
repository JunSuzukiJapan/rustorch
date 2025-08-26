# Push ����

## �,�j�÷����

### 1. ��ï�÷�clippy + format + push	
```bash
# YyfnfJ��cWfթ����Wf�÷�
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged && \
cargo fmt --all && \
git add -A && \
git commit -m "style: Apply clippy fixes and formatting" && \
git push origin $(git branch --show-current)
```

### 2. ����÷���WjL�	
```bash
# Step 1: Clippy ��ïh���c
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged

# Step 2: թ����
cargo fmt --all

# Step 3: 	���
git diff
git status

# Step 4: ����
git add -A
git commit -m "style: Apply clippy fixes and formatting

- Resolve clippy suggestions
- Apply rustfmt formatting
- Ensure code quality standards

> Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Step 5: �÷�
git push origin $(git branch --show-current)
```

## ����թ��%գ����-�

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

## �h��ï�n�÷�

```bash
# Yyfn��ï��LWfK��÷�
cargo clippy --all-targets --features "linalg-netlib" -- -D warnings && \
cargo fmt --all --check && \
cargo test --features "linalg-netlib" && \
git add -A && \
git commit -m "chore: Code quality improvements" && \
git push origin $(git branch --show-current)
```

## ��ꢹ-��׷��	

`.bashrc` ~_o `.zshrc` k��:

```bash
alias rustpush='cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged && cargo fmt --all && git add -A && git commit -m "style: Apply clippy fixes and formatting" && git push origin $(git branch --show-current)'
```

(��:
```bash
rustpush
```

## ������ƣ�

### Clippy ���L���cgMjD4
```bash
# fJ���
cargo clippy --all-targets --features "linalg-netlib"

# K��c�k���L
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged
```

### թ������ïn
```bash
# ��ïn	�jW	
cargo fmt --all --check
```

### y�ա��n�
```bash
# y�ա��nթ����
cargo fmt -- src/lib.rs src/tensor/mod.rs

# y�ա��n��Wf����
git add src/lib.rs src/tensor/mod.rs
git commit -m "style: Format specific modules"
git push origin $(git branch --show-current)
```