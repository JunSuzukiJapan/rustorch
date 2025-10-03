# Configuration Guide

RusTorch CLI supports persistent configuration through TOML files, allowing you to set default values for model paths, generation parameters, backends, and UI preferences.

## Configuration Priority

Settings are applied in the following order (later values override earlier ones):

1. **Default values** (hardcoded in application)
2. **Configuration file** (`~/.rustorch/config.toml`)
3. **Environment variables** (reserved for future use)
4. **Command-line arguments** (highest priority)

## Configuration File Location

**Default location**: `~/.rustorch/config.toml`

On macOS/Linux:
```
~/.rustorch/config.toml
```

The configuration directory is created automatically when you save a configuration.

## Configuration File Format

The configuration file uses TOML format with the following sections:

### Complete Example

```toml
[model]
default = "models/llama-7b.gguf"
cache_dir = "~/.rustorch/models"

[generation]
max_tokens = 512
temperature = 0.7
top_p = 0.9
top_k = 40

[backend]
default = "metal"
fallback = "cpu"

[session]
auto_save = true
history_file = "~/.rustorch/history"
max_history = 1000

[ui]
color = true
stream = true
show_metrics = false
```

## Configuration Sections

### [model]

Controls default model loading behavior.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default` | String (optional) | None | Default model file path to load on startup |
| `cache_dir` | String | `~/.rustorch/models` | Directory for cached model files |

**Example**:
```toml
[model]
default = "models/llama-7b.gguf"
cache_dir = "~/my-models"
```

### [generation]

Default text generation parameters.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_tokens` | Integer | 512 | Maximum number of tokens to generate |
| `temperature` | Float | 0.7 | Sampling temperature (0.0 = deterministic, higher = more random) |
| `top_p` | Float | 0.9 | Nucleus sampling threshold (0.0-1.0) |
| `top_k` | Integer | 40 | Top-k sampling: only sample from top k tokens |

**Example**:
```toml
[generation]
max_tokens = 1024
temperature = 0.8
top_p = 0.95
top_k = 50
```

**Parameter Guidelines**:
- **Temperature**:
  - `0.0-0.3`: More focused, deterministic responses
  - `0.4-0.7`: Balanced creativity and coherence
  - `0.8-1.0`: More creative, diverse responses
  - `>1.0`: Very random, experimental

- **Top-p** (Nucleus Sampling):
  - `0.9`: Standard setting for most use cases
  - `0.95`: Slightly more diverse
  - `<0.9`: More focused on high-probability tokens

- **Top-k**:
  - `40`: Balanced diversity
  - `10-20`: More focused responses
  - `50-100`: More diverse vocabulary

### [backend]

Computation backend selection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default` | String | `"cpu"` | Primary backend to use |
| `fallback` | String | `"cpu"` | Fallback if primary backend fails |

**Available backends**:
- `cpu`: CPU computation (always available)
- `metal`: Apple Metal GPU (macOS only)
- `cuda`: NVIDIA CUDA GPU (requires CUDA support)
- `opencl`: OpenCL GPU (cross-platform)
- `hybrid`: Hybrid f64 precision mode
- `hybrid-f32`: Hybrid f32 precision mode

**Example**:
```toml
[backend]
default = "metal"
fallback = "cpu"
```

### [session]

Session and history management.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_save` | Boolean | `true` | Automatically save session on exit |
| `history_file` | String | `~/.rustorch/history` | Default history file path |
| `max_history` | Integer | 1000 | Maximum number of history entries to keep |

**Example**:
```toml
[session]
auto_save = true
history_file = "~/.rustorch/my_conversations"
max_history = 5000
```

### [ui]

User interface preferences.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `color` | Boolean | `true` | Enable colored terminal output |
| `stream` | Boolean | `true` | Enable streaming token display |
| `show_metrics` | Boolean | `false` | Show performance metrics (tokens/sec, etc.) |

**Example**:
```toml
[ui]
color = true
stream = true
show_metrics = true
```

## Managing Configuration

### View Current Configuration

Use the `/config` command in the REPL:

```
You> /config
╔════════════════════════════════════════════════════════════╗
║                   Configuration                            ║
╚════════════════════════════════════════════════════════════╝

  Max tokens:      512
  Temperature:     0.7
  Top-p:           0.9
  Top-k:           40
  Backend:         metal
  Model:           models/llama-7b.gguf
```

### Save Current Configuration

Save your current session settings to the configuration file:

```
You> /config save
Configuration saved to: ~/.rustorch/config.toml
```

Save to a custom location:

```
You> /config save my-config.toml
Configuration saved to: my-config.toml
```

### Create Configuration File Manually

1. Create the configuration directory:
   ```bash
   mkdir -p ~/.rustorch
   ```

2. Create `~/.rustorch/config.toml`:
   ```bash
   nano ~/.rustorch/config.toml
   ```

3. Add your desired settings (see examples above)

4. Launch RusTorch CLI - it will automatically load the configuration

### Partial Configuration

You can specify only the settings you want to override. Unspecified values will use defaults.

**Minimal example** (only override generation parameters):
```toml
[generation]
max_tokens = 256
temperature = 0.5
```

**Model-only example**:
```toml
[model]
default = "models/my-model.gguf"
```

## Command-Line Override

Command-line arguments always override configuration file values:

```bash
# Configuration file has temperature=0.7
# This command overrides to 0.9
rustorch-cli --model model.gguf --temperature 0.9
```

## Environment Variables

**Tilde (~) expansion**: The configuration system automatically expands `~` to your home directory.

```toml
[model]
cache_dir = "~/my-models"  # Expands to /Users/yourname/my-models
```

## Configuration Examples

### Example 1: Creative Writing

```toml
[generation]
max_tokens = 2048
temperature = 0.9
top_p = 0.95
top_k = 60

[model]
default = "models/creative-writer-13b.gguf"
```

### Example 2: Code Generation

```toml
[generation]
max_tokens = 1024
temperature = 0.2
top_p = 0.9
top_k = 30

[model]
default = "models/code-llama-7b.gguf"
```

### Example 3: Fast Inference (Metal GPU)

```toml
[backend]
default = "metal"
fallback = "cpu"

[generation]
max_tokens = 512
temperature = 0.7

[ui]
stream = true
show_metrics = true
```

### Example 4: Minimal Configuration

```toml
[model]
default = "models/llama-7b.gguf"

[backend]
default = "metal"
```

## Troubleshooting

### Configuration Not Loading

1. **Check file location**:
   ```bash
   ls -la ~/.rustorch/config.toml
   ```

2. **Verify TOML syntax**:
   ```bash
   # Use a TOML validator online or:
   rustorch-cli --help  # Will show errors if config is invalid
   ```

3. **Enable debug logging**:
   ```bash
   rustorch-cli --log-level debug
   ```
   Check for messages like:
   - `Loaded configuration from ~/.rustorch/config.toml`
   - `No config file found, using defaults`

### Invalid Configuration Values

If configuration values are invalid, defaults will be used:

```
WARNING: Invalid temperature value in config: 2.5 (must be >= 0.0)
Using default: 0.7
```

### Permission Errors

If you see permission errors when saving configuration:

```bash
chmod 700 ~/.rustorch
chmod 600 ~/.rustorch/config.toml
```

## Best Practices

1. **Start with minimal config**: Only override what you need
2. **Use comments**: TOML supports `#` comments for documentation
3. **Version control**: Consider adding your config to dotfile repos
4. **Test changes**: Use `/config` command to verify loaded settings
5. **Backup important configs**: Keep copies of working configurations

## See Also

- [EXAMPLES.md](EXAMPLES.md) - Usage examples with different configurations
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [REQUIREMENTS.md](REQUIREMENTS.md) - Full requirements specification
