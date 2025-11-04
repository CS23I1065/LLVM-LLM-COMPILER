import os
import shutil
import llvmlite.binding as llvm
import glob
import json
from collections import Counter

SOURCE_UNOPT_DIR = "dataset/unoptimized_ir"
SOURCE_OPT_DIR = "dataset/optimized_ir"

# Our filtered dataset will now be more organized
FILTERED_UNOPT_DIR = "dataset_filtered/unoptimized_ir"
FILTERED_OPT_DIR = "dataset_filtered/optimized_ir"
MANIFEST_PATH = "dataset_filtered/manifest.json"

# Create the new directories for our clean dataset
os.makedirs(FILTERED_UNOPT_DIR, exist_ok=True)
os.makedirs(FILTERED_OPT_DIR, exist_ok=True)

# Initialize llvmlite
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

def analyze_ir_file(ll_file_path):
    """
    Parses an LLVM IR file and returns a dictionary of structural metrics.
    
    Returns None if parsing fails.
    """
    try:
        with open(ll_file_path, 'r') as f:
            ll_ir_str = f.read()
            
        if not ll_ir_str.strip():
            return None # Skip empty files
            
        mod = llvm.parse_assembly(ll_ir_str)
        mod.verify()
        
        instr_count = 0
        block_count = 0
        opcode_counts = Counter()

        for func in mod.functions:
            if func.is_declaration:
                continue
            
            for block in func.blocks:
                block_count += 1
                
                # --- START FIX ---
                # We removed the buggy 'len(block.instructions)' line
                # and now increment the count inside the loop.
                for instr in block.instructions:
                    instr_count += 1 # This is the fix
                    opcode_counts[instr.opcode] += 1
                # --- END FIX ---
                    
        return {
            "instruction_count": instr_count,
            "basic_block_count": block_count,
            "opcode_counts": dict(opcode_counts)
        }
        
    except Exception as e:
        # Catch parsing errors, which can happen with csmith
        print(f"Warning: Failed to parse {ll_file_path}. Skipping. Error: {e}")
        return None

def get_top_opcodes(opcode_counts, k=5):
    """Helper to get top k opcodes for the manifest."""
    # Convert Counter to dict for JSON serialization
    counts_dict = dict(opcode_counts)
    # Get top k items
    top_items = sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)[:k]
    return dict(top_items)

# --- Main Analysis Loop ---
if __name__ == "__main__":
    print("Starting high-impact analysis and filtering...")
    
    manifest = {
        "kept_files": {},
        "discarded_files": {}
    }
    
    total_pairs = 0
    kept_pairs = 0
    
    unopt_files = glob.glob(os.path.join(SOURCE_UNOPT_DIR, "*_unopt.ll"))
    total_pairs = len(unopt_files)

    for unopt_path in unopt_files:
        base_name = os.path.basename(unopt_path).replace("_unopt.ll", "")
        opt_file_name = f"{base_name}_opt.ll"
        opt_path = os.path.join(SOURCE_OPT_DIR, opt_file_name)
        
        reason_discarded = ""

        if not os.path.exists(opt_path):
            reason_discarded = "MISSING_OPTIMIZED_PAIR"
            manifest["discarded_files"][base_name] = reason_discarded
            continue

        unopt_stats = analyze_ir_file(unopt_path)
        opt_stats = analyze_ir_file(opt_path)

        # --- Filtering Logic ---
        
        # 1. Sanity Check (Stage 1)
        if not unopt_stats or not opt_stats:
            reason_discarded = "PARSE_FAILURE_OR_EMPTY"
        
        # 2. Trivial Case Elimination (Stage 2)
        elif unopt_stats["instruction_count"] < 10 or unopt_stats["basic_block_count"] < 2:
            reason_discarded = "TRIVIAL_FILE_TOO_SIMPLE"
            
        # 3. No-Change Elimination (Stage 2)
        elif unopt_stats["instruction_count"] == opt_stats["instruction_count"]:
            reason_discarded = "NO_INSTRUCTION_COUNT_CHANGE"

        # --- End Logic ---

        if reason_discarded:
            manifest["discarded_files"][base_name] = reason_discarded
        else:
            # This is a "good" pair. Keep it.
            kept_pairs += 1
            
            # Copy the files
            shutil.copy(unopt_path, FILTERED_UNOPT_DIR)
            shutil.copy(opt_path, FILTERED_OPT_DIR)
            
            # Log its high-impact metrics to the manifest
            manifest["kept_files"][base_name] = {
                "unopt_stats": {
                    "instr_count": unopt_stats["instruction_count"],
                    "block_count": unopt_stats["basic_block_count"],
                    "top_opcodes": get_top_opcodes(Counter(unopt_stats["opcode_counts"]))
                },
                "opt_stats": {
                    "instr_count": opt_stats["instruction_count"],
                    "block_count": opt_stats["basic_block_count"],
                    "top_opcodes": get_top_opcodes(Counter(opt_stats["opcode_counts"]))
                }
            }

    # Write the final manifest file
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAnalysis complete.")
    print(f"Processed: {total_pairs} pairs")

    # Add a check for total_pairs to prevent ZeroDivisionError
    if total_pairs > 0:
        print(f"Kept:      {kept_pairs} pairs ({(kept_pairs/total_pairs*100):.2f}%)")
    else:
        print(f"Kept:      {kept_pairs} pairs")
        print(f"\nERROR: No files were found in '{SOURCE_UNOPT_DIR}' matching '*_unopt.ll'")
        print("Please check that the directory exists and contains your generated files.")

    print(f"Filtered dataset is ready in 'dataset_filtered/'")
    print(f"High-impact analysis log saved to '{MANIFEST_PATH}'")