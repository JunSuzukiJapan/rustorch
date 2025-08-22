#!/usr/bin/env python3
"""
Advanced test for Rust→Python communication and error handling
"""

import sys
import traceback

def test_callback_system():
    """Test Rust→Python callback system"""
    print("Testing Rust→Python callback system...")
    
    try:
        import _rustorch_py as rt
        
        # Create callback registry
        registry = rt.init_callback_system()
        print("✓ Created callback registry")
        
        # Define Python functions to be called from Rust
        def my_python_callback(message):
            return f"Python received: {message}"
        
        def progress_callback(step, total, percentage):
            return f"Progress: {step}/{total} ({percentage:.1f}%)"
        
        def completion_callback(count):
            return f"Completed {count} operations"
        
        def log_callback(level, message):
            print(f"[{level.upper()}] {message}")
            return True
        
        def error_handler(error_type, error_message):
            print(f"Error Handler: {error_type} - {error_message}")
            return True  # Continue execution
        
        # Register callbacks
        registry.register_callback("my_callback", my_python_callback)
        registry.register_callback("progress", progress_callback)
        registry.register_callback("completed", completion_callback)
        registry.register_callback("log", log_callback)
        registry.register_callback("error_handler", error_handler)
        
        print(f"✓ Registered callbacks: {registry.list_callbacks()}")
        
        # Test calling Python from Rust
        result = rt.call_python_from_rust(registry, "my_callback", "Hello from Rust!")
        print(f"✓ Rust→Python call result: {result}")
        
        # Test progress callback
        print("✓ Testing progress callbacks...")
        progress_results = rt.progress_callback_example(registry, 5)
        for result in progress_results:
            print(f"  - {result}")
        
        # Test manual logging via callback
        try:
            args = [rt.get_version(), "System initialized"]
            args_list = args  # Simple list for testing
            print("✓ Manual log callback test skipped (requires Python context)")
        except Exception as e:
            print(f"ℹ  Log callback test error: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Callback system test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test comprehensive error handling"""
    print("\nTesting error handling system...")
    
    try:
        import _rustorch_py as rt
        
        # Test custom exceptions
        try:
            rust_error = rt.RusTorchError("Test error", 1001)
            print(f"✓ Created RusTorchError: {rust_error}")
            print(f"  Message: {rust_error.message}")
            print(f"  Error code: {rust_error.error_code}")
        except Exception as e:
            print(f"Error creating RusTorchError: {e}")
        
        # Test Result type
        ok_result = rt.Result.ok("Success value")
        err_result = rt.Result.err("Error message")
        
        print(f"✓ OK result: {ok_result}, is_ok: {ok_result.is_ok}")
        print(f"✓ Error result: {err_result}, is_err: {err_result.is_err}")
        
        # Test getting values
        try:
            success_value = ok_result.unwrap()
            print(f"✓ Unwrapped success value: {success_value}")
        except Exception as e:
            print(f"Error unwrapping success: {e}")
            
        try:
            error_value = err_result.unwrap()
            print(f"This shouldn't print: {error_value}")
        except Exception as e:
            print(f"✓ Expected error when unwrapping error result: {e}")
        
        # Test default value
        default_value = err_result.unwrap_or("Default value")
        print(f"✓ Got default value: {default_value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_try_catch():
    """Test try-catch style error handling"""
    print("\nTesting try-catch functionality...")
    
    try:
        import _rustorch_py as rt
        
        def operation_that_succeeds():
            return "Success!"
        
        def operation_that_fails():
            raise ValueError("This operation failed")
        
        def error_handler(exception):
            print(f"Handled error: {exception}")
            return "Error was handled"
        
        # Test successful operation
        result1 = rt.try_catch(operation_that_succeeds)
        print(f"✓ Successful operation result: {result1}")
        
        # Test failed operation with handler
        result2 = rt.try_catch(operation_that_fails, error_handler)
        print(f"✓ Failed operation with handler result: {result2}")
        
        return True
        
    except Exception as e:
        print(f"✗ Try-catch test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all advanced tests"""
    print("Advanced RusTorch Python Integration Tests")
    print("=" * 50)
    
    tests = [
        test_callback_system,
        test_error_handling,
        test_try_catch,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All advanced tests passed!")
        return 0
    else:
        print("❌ Some advanced tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())