#!/usr/bin/env python3
"""
Ultra-Simple CUDA Test
======================

This script tests the most basic CUDA functionality to isolate issues.
Uses only subprocess to call nvidia-smi and nvcc.

No external Python packages required.
"""

import subprocess
import sys
import platform

def test_nvidia_driver():
    """Test NVIDIA driver with nvidia-smi"""
    print("🔍 Testing NVIDIA Driver")
    print("=" * 25)
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode == 0:
            print("✅ NVIDIA driver working")
            
            # Extract basic info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    driver = line.split('Driver Version: ')[1].split()[0]
                    print(f"   📦 Driver: {driver}")
                if 'CUDA Version' in line:
                    cuda = line.split('CUDA Version: ')[1].split()[0]
                    print(f"   🚀 CUDA: {cuda}")
            
            return True
        else:
            print("❌ nvidia-smi failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("   NVIDIA driver not installed")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timeout")
        return False
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
        return False

def test_cuda_compiler():
    """Test CUDA compiler (nvcc)"""
    print("\n🔨 Testing CUDA Compiler")
    print("=" * 25)
    
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode == 0:
            print("✅ CUDA compiler working")
            
            # Extract version
            output = result.stdout
            if 'release' in output:
                version_line = [line for line in output.split('\n') if 'release' in line][0]
                version = version_line.split('release ')[1].split(',')[0]
                print(f"   📦 NVCC version: {version}")
            
            return True
        else:
            print("❌ nvcc failed")
            return False
            
    except FileNotFoundError:
        print("❌ nvcc not found")
        print("   CUDA Toolkit not installed or not in PATH")
        return False
    except Exception as e:
        print(f"❌ nvcc error: {e}")
        return False

def test_simple_cuda_compile():
    """Test compiling a simple CUDA program"""
    print("\n⚙️  Testing CUDA Compilation")
    print("=" * 27)
    
    # Simple CUDA program
    cuda_code = '''
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d!\\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\\n");
    
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA devices found: %d\\n", deviceCount);
    
    if (deviceCount > 0) {
        // Launch kernel with 5 threads
        hello_kernel<<<1, 5>>>();
        cudaDeviceSynchronize();
        printf("Kernel completed!\\n");
    }
    
    return 0;
}
'''
    
    try:
        # Write CUDA file
        cuda_file = 'test_cuda.cu'
        exe_file = 'test_cuda.exe' if platform.system() == 'Windows' else 'test_cuda'
        
        with open(cuda_file, 'w') as f:
            f.write(cuda_code)
        
        print(f"📝 Created {cuda_file}")
        
        # Compile
        print("🔨 Compiling CUDA program...")
        compile_cmd = ['nvcc', cuda_file, '-o', exe_file]
        
        result = subprocess.run(compile_cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=30)
        
        if result.returncode == 0:
            print("✅ Compilation successful")
            
            # Try to run
            print("🚀 Running CUDA program...")
            run_result = subprocess.run([f'./{exe_file}' if platform.system() != 'Windows' else exe_file], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=10)
            
            if run_result.returncode == 0:
                print("✅ Execution successful")
                print("📤 Output:")
                for line in run_result.stdout.split('\n'):
                    if line.strip():
                        print(f"   {line}")
                return True
            else:
                print("❌ Execution failed")
                print(f"   Error: {run_result.stderr}")
                return False
        else:
            print("❌ Compilation failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Compilation test error: {e}")
        return False
    finally:
        # Clean up
        try:
            import os
            if os.path.exists(cuda_file):
                os.remove(cuda_file)
            if os.path.exists(exe_file):
                os.remove(exe_file)
        except:
            pass

def test_python_packages():
    """Test Python CUDA packages availability"""
    print("\n📦 Testing Python CUDA Packages")
    print("=" * 32)
    
    packages = [
        ('numba', 'Numba JIT compiler'),
        ('cupy', 'NumPy-like GPU arrays'),
        ('pycuda', 'Low-level CUDA bindings'),
        ('torch', 'PyTorch with CUDA'),
    ]
    
    available = []
    
    for package, description in packages:
        try:
            __import__(package)
            print(f"✅ {package:8} - {description}")
            available.append(package)
        except ImportError:
            print(f"❌ {package:8} - {description} (not installed)")
    
    return available

def main():
    """Main diagnostic function"""
    print("🩺 CUDA Diagnostic Tool")
    print("======================")
    print(f"🖥️  OS: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    all_good = True
    
    # Test 1: NVIDIA driver
    if not test_nvidia_driver():
        all_good = False
        print("\n🚨 CRITICAL: NVIDIA driver issue")
        print("   1. Install latest NVIDIA driver")
        print("   2. Restart computer")
        return
    
    # Test 2: CUDA compiler
    cuda_compiler_ok = test_cuda_compiler()
    if not cuda_compiler_ok:
        all_good = False
        print("\n⚠️  WARNING: CUDA Toolkit not properly installed")
        print("   1. Download CUDA Toolkit from nvidia.com")
        print("   2. Add to PATH")
    
    # Test 3: Simple compilation (only if nvcc works)
    if cuda_compiler_ok:
        if not test_simple_cuda_compile():
            all_good = False
            print("\n⚠️  WARNING: CUDA compilation issues")
    
    # Test 4: Python packages
    available_packages = test_python_packages()
    
    # Summary
    print("\n📋 SUMMARY")
    print("=" * 10)
    
    if all_good:
        print("🎉 All basic CUDA tests passed!")
        print("   Your CUDA installation is working")
        
        if available_packages:
            print(f"\n📦 Available Python packages: {', '.join(available_packages)}")
            print("   Try using these in your notebook")
        else:
            print("\n💡 Recommended Python packages to install:")
            print("   pip install numba")
            print("   pip install pycuda")
    else:
        print("❌ Some CUDA tests failed")
        print("   Check NVIDIA driver and CUDA Toolkit installation")

if __name__ == "__main__":
    main()