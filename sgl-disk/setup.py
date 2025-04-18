# setup.py
import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# --- Configuration ---

# 1. Define the name for your Python extension module
#    This is what you will 'import' in Python
extension_name = "rocksdb_storage_backend"

# 2. List your C++ source files
cpp_source_files = ["rocksdb.cpp"]

# 3. Specify the paths to your RocksDB installation
#    It's good practice to use environment variables, but we can hardcode
#    your provided paths as a default.
rocksdb_base_dir = os.environ.get('ROCKSDB_DIR', '/data/user_data/yuweia/rocksdb/build')
rocksdb_include_dir = os.path.join(rocksdb_base_dir, 'include')
rocksdb_lib_dir = os.path.join(rocksdb_base_dir, 'lib')

# 4. Check if the RocksDB paths exist (optional but recommended)
if not os.path.isdir(rocksdb_include_dir):
    raise RuntimeError(f"RocksDB include directory not found: {rocksdb_include_dir}\n"
                       "Make sure the path is correct or set the ROCKSDB_DIR environment variable.")
if not os.path.isdir(rocksdb_lib_dir):
    raise RuntimeError(f"RocksDB library directory not found: {rocksdb_lib_dir}\n"
                       "Make sure the path is correct or set the ROCKSDB_DIR environment variable.")

print(f"--- Using RocksDB ---")
print(f"Include path: {rocksdb_include_dir}")
print(f"Library path: {rocksdb_lib_dir}")
print(f"--------------------")


# --- Build Definition ---

setup(
    name=extension_name, # The name of the package
    version="0.1.0",     # Optional: Add a version
    author="Your Name",  # Optional: Add your name
    description="PyTorch extension for storing tensor data in RocksDB", # Optional
    ext_modules=[
        CppExtension(
            name=extension_name, # The name of the module to build (importable name)
            sources=cpp_source_files,
            # Add RocksDB include directory
            include_dirs=[
                rocksdb_include_dir,
                # PyTorch include paths are handled automatically by CppExtension
            ],
            # Add RocksDB library directory
            library_dirs=[
                rocksdb_lib_dir,
            ],
            # Link against the RocksDB library
            # Also link against pthread, as RocksDB likely uses it.
            # If you get linker errors about missing symbols for compression
            # (like snappy, lz4, zstd, z), add them here too (e.g., 'rocksdb', 'pthread', 'snappy', 'lz4')
            # depending on how your RocksDB was built.
            libraries=[
                'rocksdb',
                'pthread',
                # 'z', 'snappy', 'lz4', 'zstd' # <-- Add if needed
            ],
            # Specify compiler arguments
            extra_compile_args={
                'cxx': [
                    '-std=c++17', # Use C++17 (as required by std::make_unique and good practice)
                    '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '-O3',        # Optimization level
                    '-Wall',      # Enable common warnings
                    '-Wextra',    # Enable extra warnings
                    '-fPIC',      # Position Independent Code (required for shared libs)
                    '-DNDEBUG',   # Disable asserts in release builds (optional)
                    # Add any other flags you need
                ]
            },
            # Specify linker arguments
            extra_link_args=[
                # Embed the RPATH to the RocksDB library directory.
                # This helps the Python extension find librocksdb.so at runtime
                # without needing LD_LIBRARY_PATH.
                f'-Wl,-rpath,{rocksdb_lib_dir}'
            ]
        )
    ],
    # Use the PyTorch build extension command class
    cmdclass={
        'build_ext': BuildExtension
    }
)

print("\n--- Build Script Finished ---")
print("To compile and install:")
print("  pip install .")
print("Or for development (builds in place):")
print("  python setup.py build_ext --inplace")
print("or")
print("  pip install -e .")