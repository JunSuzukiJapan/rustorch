# RusTorch Python Integration - Usage Examples

Complete examples demonstrating Rust↔Python communication and error handling.

## Basic Usage

### 1. Simple Function Calls

```python
import _rustorch_py as rt

# Basic functions
print(rt.hello_from_rust())  # "Hello from RusTorch!"
print(rt.get_version())      # "0.3.3"
print(rt.add_numbers(1.5, 2.5))  # 4.0

# List processing
numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
total = rt.sum_list(numbers)
print(f"Sum: {total}")  # Sum: 15.0
```

## Rust→Python Communication

### 2. Setting Up Callbacks

```python
import _rustorch_py as rt

# Create callback registry
registry = rt.init_callback_system()

# Define Python functions to be called from Rust
def my_callback(message):
    return f"Python processed: {message}"

def progress_handler(step, total, percentage):
    print(f"Progress: {step}/{total} ({percentage:.1f}%)")
    return f"Step {step} completed"

def completion_handler(count):
    print(f"🎉 All {count} operations completed!")
    return "Success"

# Register callbacks
registry.register_callback("my_callback", my_callback)
registry.register_callback("progress", progress_handler)
registry.register_callback("completed", completion_handler)

print(f"Registered callbacks: {registry.list_callbacks()}")
```

### 3. Calling Python from Rust

```python
# Call Python function from Rust
result = rt.call_python_from_rust(registry, "my_callback", "Hello from Rust!")
print(result)  # "Python processed: Hello from Rust!"

# Progress callback example
print("Running progress example...")
results = rt.progress_callback_example(registry, 5)
for result in results:
    print(f"  → {result}")
```

### 4. Advanced Callback Examples

```python
# Logging callback
def log_handler(level, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level.upper()}: {message}")
    return True

# Error handling callback
def error_handler(error_type, error_message):
    print(f"🚨 Error: {error_type} - {error_message}")
    # Return True to continue, False to stop
    return True

# Data processing callback
def data_processor(data_chunk):
    # Process data from Rust
    processed = [x * 2 for x in data_chunk]
    return processed

registry.register_callback("log", log_handler)
registry.register_callback("error_handler", error_handler)
registry.register_callback("process_data", data_processor)
```

## Error Handling

### 5. Custom Exceptions

```python
import _rustorch_py as rt

# Create custom RusTorch errors
try:
    error = rt.RusTorchError("Something went wrong", 2001)
    print(f"Error: {error}")
    print(f"Message: {error.message}")
    print(f"Code: {error.error_code}")
except Exception as e:
    print(f"Failed to create error: {e}")
```

### 6. Result Type Pattern

```python
# Success case
success_result = rt.Result.ok("Operation successful")
print(f"Success: {success_result.is_ok}")  # True

if success_result.is_ok:
    value = success_result.unwrap()
    print(f"Value: {value}")

# Error case
error_result = rt.Result.err("Operation failed")
print(f"Error: {error_result.is_err}")  # True

# Safe unwrapping with default
default_value = error_result.unwrap_or("Default value")
print(f"Default: {default_value}")

# Error unwrapping (raises exception)
try:
    error_result.unwrap()
except Exception as e:
    print(f"Expected error: {e}")
```

### 7. Try-Catch Style Error Handling

```python
def risky_operation():
    # This might fail
    if random.random() < 0.5:
        raise ValueError("Random failure")
    return "Success!"

def error_handler(exception):
    print(f"Handled: {exception}")
    return "Recovered from error"

# Safe execution
result = rt.try_catch(risky_operation, error_handler)
print(f"Result: {result}")
```

## Complete Example: Neural Network Training with Callbacks

```python
import _rustorch_py as rt
import time

class TrainingCallbacks:
    def __init__(self):
        self.epoch_times = []
        self.losses = []
    
    def on_epoch_start(self, epoch, total_epochs):
        self.start_time = time.time()
        print(f"Epoch {epoch+1}/{total_epochs} started...")
        return True
    
    def on_batch_complete(self, batch, total_batches, loss):
        self.losses.append(loss)
        if batch % 10 == 0:  # Log every 10 batches
            print(f"  Batch {batch}/{total_batches}, Loss: {loss:.4f}")
        return True
    
    def on_epoch_complete(self, epoch, avg_loss):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        return True
    
    def on_training_complete(self, total_time):
        print(f"🎉 Training completed in {total_time:.2f}s")
        print(f"Average loss: {sum(self.losses)/len(self.losses):.4f}")
        return True

# Setup training
registry = rt.init_callback_system()
callbacks = TrainingCallbacks()

# Register callbacks
registry.register_callback("epoch_start", callbacks.on_epoch_start)
registry.register_callback("batch_complete", callbacks.on_batch_complete)
registry.register_callback("epoch_complete", callbacks.on_epoch_complete)
registry.register_callback("training_complete", callbacks.on_training_complete)

# Simulate training (this would be implemented in Rust)
print("Starting neural network training simulation...")
# training_result = rt.train_model(registry, model_config, dataset)
```

## Real-world Integration Patterns

### 8. Logging Integration

```python
import logging

# Setup Python logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rustorch")

def rust_logger(level, message):
    """Route Rust logs to Python logging system"""
    getattr(logger, level.lower(), logger.info)(f"[Rust] {message}")
    return True

registry.register_callback("log", rust_logger)
```

### 9. Progress Monitoring

```python
from tqdm import tqdm

class ProgressMonitor:
    def __init__(self):
        self.pbar = None
    
    def start_progress(self, total, description="Processing"):
        self.pbar = tqdm(total=total, desc=description)
        return True
    
    def update_progress(self, step, total, percentage):
        if self.pbar:
            self.pbar.update(1)
        return True
    
    def finish_progress(self, result):
        if self.pbar:
            self.pbar.close()
        print(f"Completed: {result}")
        return True

# Usage
monitor = ProgressMonitor()
registry.register_callback("start_progress", monitor.start_progress)
registry.register_callback("progress", monitor.update_progress)
registry.register_callback("completed", monitor.finish_progress)
```

### 10. Error Recovery

```python
class ErrorRecoverySystem:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.retry_count = 0
    
    def handle_error(self, error_type, error_message):
        print(f"Error occurred: {error_type} - {error_message}")
        
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            print(f"Retrying... ({self.retry_count}/{self.max_retries})")
            return True  # Continue execution
        else:
            print("Max retries reached, stopping...")
            return False  # Stop execution
    
    def reset(self):
        self.retry_count = 0

# Usage
recovery = ErrorRecoverySystem()
registry.register_callback("error_handler", recovery.handle_error)
```

## Performance Tips

1. **Minimize callback overhead**: Only register necessary callbacks
2. **Batch operations**: Process multiple items in single callback when possible
3. **Error handling**: Use Result types for expected failures
4. **Memory management**: Let PyO3 handle memory automatically

## Debugging

```python
# Enable detailed error reporting
import _rustorch_py as rt

# Check available functions
print("Available functions:", [name for name in dir(rt) if not name.startswith('_')])

# Test callback registration
registry = rt.init_callback_system()
print(f"Registry created: {registry}")

# Test basic communication
try:
    result = rt.call_python_from_rust(registry, "non_existent", "test")
except Exception as e:
    print(f"Expected error: {e}")
```

This demonstrates the full range of Rust↔Python communication capabilities, from basic function calls to sophisticated callback systems and error handling patterns.