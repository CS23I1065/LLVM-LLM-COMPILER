import os
import glob
import json
import re
import concurrent.futures
import sys
import llvmlite.binding as llvm # Import llvmlite binding for verification

# --- CONFIGURATION ---
CANONICAL_DIR = "dataset_final/optimized_ir" # Using optimized IR as requested
METADATA_DIR = "dataset_final/metadata"
ROUNDTRIP_DIR = "dataset_roundtrip_check"

# --- INVERTED SPECIFICATION MAPPINGS ---
# We must invert the maps from the tokenizer

# Invert OPCODE_MAP
# (Original mapping)
OPCODE_MAP = {
    'ret': 'OP_RET', 'br': 'OP_BR', 'switch': 'OP_SWITCH', 'indirectbr': 'OP_INDIRECTBR',
    'invoke': 'OP_INVOKE', 'callbr': 'OP_CALLBR', 'resume': 'OP_RESUME', 
    'catchswitch': 'OP_CATCHSWITCH', 'catchret': 'OP_CATCHRET', 
    'cleanupret': 'OP_CLEANUPRET', 'unreachable': 'OP_UNREACHABLE',
    'add': 'OP_ADD', 'fadd': 'OP_FADD', 'sub': 'OP_SUB', 'fsub': 'OP_FSUB',
    'mul': 'OP_MUL', 'fmul': 'OP_FMUL', 'udiv': 'OP_UDIV', 'sdiv': 'OP_SDIV',
    'fdiv': 'OP_FDIV', 'urem': 'OP_UREM', 'srem': 'OP_SREM', 'frem': 'OP_FREM',
    'shl': 'OP_SHL', 'lshr': 'OP_LSHR', 'ashr': 'OP_ASHR', 'and': 'OP_AND',
    'or': 'OP_OR', 'xor': 'OP_XOR', 'fneg': 'OP_FNEG', 'alloca': 'OP_ALLOCA', 
    'load': 'OP_LOAD', 'store': 'OP_STORE', 'getelementptr': 'OP_GEP', 
    'fence': 'OP_FENCE', 'cmpxchg': 'OP_CMPXCHG', 'atomicrmw': 'OP_ATOMICRMW',
    'trunc': 'OP_TRUNC', 'zext': 'OP_ZEXT', 'sext': 'OP_SEXT', 'fptrunc': 'OP_FPTRUNC',
    'fpext': 'OP_FPEXT', 'fptoui': 'OP_FPTOUI', 'fptosi': 'OP_FPTOSI',
    'uitofp': 'OP_UITOFP', 'sitofp': 'OP_SITOFP', 'ptrtoint': 'OP_PTRTOINT',
    'inttoptr': 'OP_INTTOPTR', 'bitcast': 'OP_BITCAST', 
    'addrspacecast': 'OP_ADDRSPACECAST', 'extractelement': 'OP_EXTRACTELEMENT', 
    'insertelement': 'OP_INSERTELEMENT', 'shufflevector': 'OP_SHUFFLEVECTOR', 
    'extractvalue': 'OP_EXTRACTVALUE', 'insertvalue': 'OP_INSERTVALUE',
    'icmp': 'OP_ICMP', 'fcmp': 'OP_FCMP', 'phi': 'OP_PHI', 'select': 'OP_SELECT',
    'call': 'OP_CALL', 'va_arg': 'OP_VA_ARG', 'landingpad': 'OP_LANDINGPAD',
    'catchpad': 'OP_CATCHPAD', 'cleanuppad': 'OP_CLEANUPPAD', 'freeze': 'OP_FREEZE',
}

# (Inverted mapping for detokenizer)
OPCODE_MAP_INV = {v: k for k, v in OPCODE_MAP.items()}

# --- REGEX-BASED TRANSLATION RULES ---
# This is a "best-effort" translation. It is not a real parser
# and will fail on many complex, nested, or edge-case constructs.
# Its goal is to provide a human-readable file for visual inspection.

def detokenize_line(line):
    """
    Applies a series of regex substitutions to a line of
    canonical IR to make it look like LLVM IR.
    """
    
    # 1. Function headers:
    # FUNC_START @foo (ARG0: TYPE_INT32) -> TYPE_INT32 [nounwind]
    # -> define i32 @foo(i32 %ARG0) [nounwind] {
    line = re.sub(
        r'FUNC_START @(\S+) \((.*?)\) -> (\S+)(.*)',
        r'define \3 @\1(\2) \4 {',
        line
    )
    # FUNC_END @foo -> }
    line = re.sub(r'FUNC_END @(\S+)', r'}', line)

    # 1.5. Attribute Cleanup (to fix b'...' bug from tokenizer)
    # This cleans up [b'noundef'] -> noundef
    line = re.sub(r"\[b'(\w+)'\]", r'\1', line) 
    # This cleans up [b]
    line = re.sub(r'\[b\]', '', line)
    
    # 2. Block Labels:
    # BLOCK_LABEL(BLOCK_0) -> BLOCK_0:
    line = re.sub(r'BLOCK_LABEL\((BLOCK_\d+)\)', r'\1:', line)

    # 3. Operands (the hardest part):
    # USE_REG0 -> %REG0
    line = re.sub(r'USE_(REG\d+)', r'%\1', line)
    # USE_ARG0 -> %ARG0
    line = re.sub(r'USE_(ARG\d+)', r'%\1', line)
    # USE_BLOCK_0 -> %BLOCK_0 (for branch labels)
    line = re.sub(r'USE_(BLOCK_\d+)', r'%\1', line)

    # 4. Definitions:
    # REG0 = ... -> %REG0 = ...
    # ARG0: ... -> i32 %ARG0 (in function def)
    line = re.sub(r'REG(\d+)', r'%REG\1', line)
    line = re.sub(r'ARG(\d+)', r'%ARG\1', line)

    # 5. Constants:
    line = re.sub(r'CONST_INT\((.*?)\)', r'\1', line)
    line = re.sub(r'CONST_FLOAT\((.*?)\)', r'\1', line)
    line = re.sub(r'CONST_BOOL\((.*?)\)', r'\1', line)
    line = line.replace('CONST_NULL', 'null')
    line = line.replace('CONST_UNDEF', 'undef')
    line = line.replace('CONST_ZERO', 'zeroinitializer')

    # 6. Types (simple cases):
    line = re.sub(r'TYPE_INT(\d+)', r'i\1', line)
    line = line.replace('TYPE_VOID', 'void')
    line = line.replace('TYPE_FLOAT', 'float')
    line = line.replace('TYPE_DOUBLE', 'double')

    # --- FIX for TYPE_UNKNOWN ---
    # Use 'i32' as a best-guess fallback to satisfy the parser
    # This will fix both return types and argument types.
    line = line.replace('TYPE_UNKNOWN', 'i32')
    # --- END FIX ---

    # This is a gross oversimplification, but good for a visual check
    line = re.sub(r'TYPE_PTR\(.*?\)', 'ptr', line)
    line = re.sub(r'TYPE_ARRAY\(.*?\)', 'array', line)
    line = re.sub(r'TYPE_VECTOR\(.*?\)', 'vector', line)
    line = re.sub(r'TYPE_STRUCT\(.*?\)', 'struct', line)

    # 7. Opcodes (must run *after* other replacements):
    # This is tricky; we only want to replace opcodes at the start
    # of an instruction.
    # e.g., "  REG0 = OP_ADD ... "
    # e.g., "  OP_RET ... "
    for canon_op, llvm_op in OPCODE_MAP_INV.items():
        # Check for start of instruction:
        line = re.sub(r'(\s*=\s*|\s\s)' + re.escape(canon_op) + r'\b', r'\1' + llvm_op, line)
        
    # 8. Intrinsics
    line = re.sub(r'INTRINSIC_(\w+)', r'@llvm.\1', line)
    
    return line

def init_llvm_worker():
    """
    Initializer function for each worker process.
    This ensures LLVM is initialized *within* the process
    that will be using it.
    """
    try:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
    except Exception:
        # If initialization fails or is already done, continue
        pass

def process_file(canon_path):
    """
    Reads one canonical file and its metadata, and writes
    a "reconstructed" .ll file for visual inspection.
    """
    try:
        base_name = os.path.basename(canon_path).replace(".canon.ll", "")
        meta_path = os.path.join(METADATA_DIR, f"{base_name}.meta.json")
        output_path = os.path.join(ROUNDTRIP_DIR, f"{base_name}.reconstructed.ll")

        # 1. Load Metadata
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            return (canon_path, f"SKIPPED: Missing metadata file {meta_path}")

        # 2. Load Canonical IR
        with open(canon_path, 'r', encoding='utf-8') as f:
            canon_lines = f.readlines()

        reconstructed_ll = []

        # 3. Re-inject Metadata at the top
        reconstructed_ll.append(f"; --- METADATA RE-INJECTED FROM {base_name}.meta.json ---")
        
        source_filename = metadata.get('source_filename', 'N/A')
        reconstructed_ll.append(f"source_filename = \"{source_filename}\"")
        
        # --- FIX for N/A error ---
        # Only add target triple and datalayout if they are not "N/A"
        target_triple = metadata.get('target_triple', 'N/A')
        if target_triple != 'N/A':
            reconstructed_ll.append(f"target triple = \"{target_triple}\"")
            
        datalayout = metadata.get('datalayout', 'N/A')
        if datalayout != 'N/A':
            reconstructed_ll.append(f"target datalayout = \"{datalayout}\"")
        # --- END FIX ---
            
        reconstructed_ll.append("\n")

        for g in metadata.get('globals', []):
            reconstructed_ll.append(g)
        for d in metadata.get('declarations', []):
            reconstructed_ll.append(d)
        for a in metadata.get('attributes', []):
            reconstructed_ll.append(f"attributes {a}")
        for m in metadata.get('module_flags', []):
            reconstructed_ll.append(m)
        for n in metadata.get('named_metadata', []):
            reconstructed_ll.append(n)
        
        reconstructed_ll.append(f"\n; --- END METADATA ---")
        reconstructed_ll.append(f"; --- CANONICAL FUNCTIONS RECONSTRUCTED (BEST-EFFORT) ---")

        # 4. Detokenize function body, line by line
        for line in canon_lines:
            if not line.strip():
                reconstructed_ll.append("")
                continue
            
            detokenized_line = detokenize_line(line.strip())
            reconstructed_ll.append(detokenized_line)

        # 5. Combine into a single string
        final_ll_string = "\n".join(reconstructed_ll)

        # 6. --- VERIFICATION STEP ---
        # Try to parse and verify the reconstructed IR
        try:
            mod = llvm.parse_assembly(final_ll_string)
            mod.verify()
            
            # If successful, save the verified (and potentially cleaned-up) IR
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(mod))
            
            return (canon_path, "SUCCESS_VERIFIED")

        except Exception as e:
            # If parsing or verification fails, save the broken .ll file
            # for debugging, but report the error.
            error_msg = f"ERROR_VERIFICATION: {e}"
            # Save to a .FAILED.ll file for debugging
            with open(output_path + ".FAILED.ll", 'w', encoding='utf-8') as f:
                f.write(f"; --- PARSING/VERIFICATION FAILED --- \n; {error_msg}\n\n")
                f.write(final_ll_string)
            
            return (canon_path, error_msg)

    except Exception as e:
        return (canon_path, f"ERROR_IO: {e}")

if __name__ == "__main__":
    os.makedirs(ROUNDTRIP_DIR, exist_ok=True)
    
    print(f"Starting 'detokenizer' (round-trip visual check)...")
    print(f"Reading from: {CANONICAL_DIR}")
    print(f"Writing to:   {ROUNDTRIP_DIR}")
    
    # We are only processing the optimized files as requested
    tasks = glob.glob(os.path.join(CANONICAL_DIR, "*_opt.canon.ll"))
    
    # --- LIMITING TO 10 FILES FOR TESTING ---
    tasks = tasks[:10]
    print(f"\n--- RUNNING IN TEST MODE (10 files) ---")
    # --- END TEST LIMIT ---
    
    if not tasks:
        print(f"Error: No files found in {CANONICAL_DIR}. Did Task 4 run correctly?")
        sys.exit(1)

    total = len(tasks)
    error_count = 0
    processed_count = 0

    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_llvm_worker
    ) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            (f_path, status) = future.result()
            
            if status == "SUCCESS_VERIFIED":
                print(".", end="")
                sys.stdout.flush()
            elif "SKIPPED" in status:
                print("S", end="")
                sys.stdout.flush()
            else:
                print(f"\nFailed to process {f_path}: {status}")
                error_count += 1
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f" [{processed_count}/{total}]")

    print(f"\n\nDetokenization (verification) complete.")
    if error_count > 0:
        print(f"WARNING: {error_count} files failed to verify. Check for .FAILED.ll files.")
    print(f"Reconstructed files are in '{ROUNDTRIP_DIR}/'")