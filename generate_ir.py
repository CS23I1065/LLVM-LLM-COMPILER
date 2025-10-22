import os
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

SOURCE_DIR = "dataset/raw_c"
UNOPTIMIZED_DIR = "dataset/unoptimized_ir"
OPTIMIZED_DIR = "dataset/optimized_ir"

# Define the worker function that processes a single C file
def generate_ir(c_file_path):
    """
    Takes one C file and generates both -O0 and -O2 LLVM IR files
    with _unopt and _opt suffixes.
    """
    base_name = os.path.basename(c_file_path)
    # file_name_no_ext will be '0', '1', '2', etc.
    file_name_no_ext = os.path.splitext(base_name)[0]
    
    unopt_ll_name = f"{file_name_no_ext}_unopt.ll"
    opt_ll_name = f"{file_name_no_ext}_opt.ll"

    unoptimized_path = os.path.join(UNOPTIMIZED_DIR, unopt_ll_name)
    optimized_path = os.path.join(OPTIMIZED_DIR, opt_ll_name)

    try:
        # 1. Generate Unoptimized (-O0) IR
        #    -I/usr/include/csmith flag is ADDED
        subprocess.run(
            ["clang", "-S", "-emit-llvm", "-O0", "-I/usr/include/csmith", c_file_path, "-o", unoptimized_path],
            timeout=30, check=True, capture_output=True
        )

        # 2. Generate Optimized (-O2) IR
        #    -I/usr/include/csmith flag is ADDED
        subprocess.run(
            ["clang", "-S", "-emit-llvm", "-O2", "-I/usr/include/csmith", c_file_path, "-o", optimized_path],
            timeout=30, check=True, capture_output=True
        )

        # Updated return message to be more descriptive
        return f"Successfully processed {base_name} -> {unopt_ll_name} and {opt_ll_name}"

    except subprocess.CalledProcessError as e:
        return f"Failed (clang error) on {base_name}: {e.stderr.decode()}"
    except subprocess.TimeoutExpired:
        return f"Failed (timeout) on {base_name}"
    except Exception as e:
        return f"Failed (unknown error) on {base_name}: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    c_files = glob.glob(os.path.join(SOURCE_DIR, "*.c"))

    # Use a reasonable number of workers. 8 is a safe starting point.
    # This will run 8 clang processes at a time.
    MAX_WORKERS = 8 

    print(f"Starting IR generation for {len(c_files)} files...")

    # This is the parallelization part
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs to the pool
        futures = {executor.submit(generate_ir, f): f for f in c_files}

        # Collect results as they complete
        for future in as_completed(futures):
            print(future.result())

    print("IR generation complete.")
