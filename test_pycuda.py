#!/usr/bin/env python3
"""
Simple PyCUDA Test Script
========================

This script tests basic PyCUDA functionality with minimal code.
If this works, PyCUDA might be a better alternative to Numba CUDA.

Installation:
    pip install pycuda

Requirements:
    - NVIDIA GPU
    - CUDA Toolkit
    - PyCUDA package
"""

import sys
import time
import numpy as np

def test_pycuda_basic():
    """Test basic PyCUDA import and device detection"""
    print("🧪 Testing PyCUDA Basic Functionality")
    print("=" * 40)
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # Automatically initializes CUDA
        from pycuda.compiler import SourceModule
        
        print("✅ PyCUDA imported successfully")
        
        # Get device info
        device = cuda.Device(0)
        print(f"📱 GPU: {device.name()}")
        print(f"💾 Memory: {device.total_memory() // (1024**3)} GB")
        print(f"🔧 Compute Capability: {device.compute_capability()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyCUDA not installed: {e}")
        print("   Install with: pip install pycuda")
        return False
    except Exception as e:
        print(f"❌ PyCUDA error: {e}")
        return False

def test_simple_kernel():
    """Test a simple CUDA kernel with PyCUDA"""
    print("\n🚀 Testing Simple CUDA Kernel")
    print("=" * 30)
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import pycuda.gpuarray as gpuarray
        
        # Simple kernel code (C-like CUDA)
        kernel_code = """
        __global__ void double_array(float *a, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                a[idx] = a[idx] * 2.0f;
            }
        }
        """
        
        # Compile the kernel
        print("🔨 Compiling CUDA kernel...")
        mod = SourceModule(kernel_code)
        double_func = mod.get_function("double_array")
        
        # Create test data
        n = 10
        data_cpu = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
        print(f"📊 Input data:  {data_cpu}")
        
        # Allocate GPU memory and copy data
        print("📤 Transferring to GPU...")
        data_gpu = gpuarray.to_gpu(data_cpu)
        
        # Launch kernel
        print("⚡ Running kernel...")
        start_time = time.perf_counter()
        
        double_func(
            data_gpu,
            np.int32(n),
            block=(32, 1, 1),  # 32 threads per block
            grid=(1, 1)        # 1 block
        )
        
        cuda.Context.synchronize()  # Wait for completion
        exec_time = time.perf_counter() - start_time
        
        # Get result back
        print("📥 Transferring from GPU...")
        result = data_gpu.get()
        
        print(f"📊 Output data: {result}")
        print(f"⏱️  Execution time: {exec_time:.6f}s")
        
        # Verify result
        expected = data_cpu * 2
        success = np.allclose(result, expected)
        print(f"✅ Test {'PASSED' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"❌ Kernel test failed: {e}")
        return False

def test_vector_add():
    """Test vector addition with timing comparison"""
    print("\n📈 Performance Test: Vector Addition")
    print("=" * 35)
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import pycuda.gpuarray as gpuarray
        
        # Vector addition kernel
        kernel_code = """
        __global__ void vector_add(float *a, float *b, float *c, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """
        
        mod = SourceModule(kernel_code)
        vector_add_func = mod.get_function("vector_add")
        
        # Test with different sizes
        sizes = [1000, 10000, 100000]
        
        for n in sizes:
            print(f"\n🔢 Testing size: {n:,} elements")
            
            # Create random data
            a_cpu = np.random.randn(n).astype(np.float32)
            b_cpu = np.random.randn(n).astype(np.float32)
            
            # CPU version
            start = time.perf_counter()
            c_cpu = a_cpu + b_cpu
            time_cpu = time.perf_counter() - start
            print(f"   💻 CPU time: {time_cpu:.6f}s")
            
            # GPU version
            start = time.perf_counter()
            
            # Transfer to GPU
            a_gpu = gpuarray.to_gpu(a_cpu)
            b_gpu = gpuarray.to_gpu(b_cpu)
            c_gpu = gpuarray.empty_like(a_gpu)
            
            # Configure grid and block
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            
            # Run kernel
            vector_add_func(
                a_gpu, b_gpu, c_gpu, np.int32(n),
                block=(threads_per_block, 1, 1),
                grid=(blocks_per_grid, 1)
            )
            
            cuda.Context.synchronize()
            c_gpu_result = c_gpu.get()
            
            time_gpu = time.perf_counter() - start
            print(f"   🎮 GPU time: {time_gpu:.6f}s")
            
            # Check accuracy
            max_error = np.max(np.abs(c_cpu - c_gpu_result))
            print(f"   📊 Max error: {max_error:.2e}")
            
            # Speedup
            if time_gpu > 0:
                speedup = time_cpu / time_gpu
                print(f"   ⚡ Speedup: {speedup:.2f}x")
            
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔬 PyCUDA Test Suite")
    print("===================")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📦 NumPy: {np.__version__}")
    
    # Test 1: Basic functionality
    if not test_pycuda_basic():
        print("\n❌ Basic test failed. Cannot proceed.")
        return
    
    # Test 2: Simple kernel
    if not test_simple_kernel():
        print("\n❌ Kernel test failed.")
        return
    
    # Test 3: Performance test
    if not test_vector_add():
        print("\n❌ Performance test failed.")
        return
    
    print("\n🎉 All PyCUDA tests passed!")
    print("\n💡 Next steps:")
    print("   • PyCUDA is working on your system")
    print("   • You can use PyCUDA instead of Numba CUDA")
    print("   • Consider using PyCUDA in your GPU notebook")

if __name__ == "__main__":
    main()