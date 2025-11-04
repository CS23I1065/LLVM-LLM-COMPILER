import os
import glob
import json
import llvmlite.binding as llvm
import re
from collections import OrderedDict
import concurrent.futures
from concurrent.futures import as_completed
import sys

# --- CONFIGURATION ---
FILTERED_UNOPT_DIR = "dataset_filtered/unoptimized_ir"
FILTERED_OPT_DIR = "dataset_filtered/optimized_ir"
CANONICAL_UNOPT_DIR = "dataset_final/unoptimized_ir"
CANONICAL_OPT_DIR = "dataset_final/optimized_ir"
METADATA_DIR = "dataset_final/metadata"

# --- SPECIFICATION MAPPINGS ---
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

# --- 1. CORE PROCESSING PIPELINE ---

def process_ir_file(ll_file_path, base_name, canonical_dir, metadata_dir):
    """
    Full processing pipeline for a single .ll file.
    Parses the IR, separates metadata, and canonicalizes functions.
    (This function is executed by a worker process)
    """
    try:
        with open(ll_file_path, 'r', encoding='utf-8', errors='replace') as f:
            ll_ir_str = f.read()
            
        if not ll_ir_str.strip():
            return (ll_file_path, "SKIPPED_EMPTY")

        # 1. Parse the entire module with llvmlite
        mod = llvm.parse_assembly(ll_ir_str)
        mod.verify()
        
        # 2. Strip and Save Metadata (Robustly)
        metadata = {
            "source_filename": getattr(mod, 'source_filename', 'N/A'),
            "target_triple": getattr(mod, 'target_triple', 'N/A'),
            "datalayout": str(getattr(mod, 'datalayout', 'N/A')),
            "globals": [str(g) for g in getattr(mod, 'global_variables', [])],
            "declarations": [str(f) for f in getattr(mod, 'functions', []) if f.is_declaration],
            "attributes": [str(attr_set) for attr_set in getattr(mod, 'attributes', [])],
            "module_flags": [f"!{k} = !{{{', '.join(str(v) for v in vs)}}}" for k, vs in getattr(mod, 'module_flags', {}).items()],
            "named_metadata": [str(nm) for nm in getattr(mod, 'named_metadata', [])]
        }
        
        meta_path = os.path.join(metadata_dir, f"{base_name}.meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        global_attr_map = {str(i): str(attr_set) for i, attr_set in enumerate(getattr(mod, 'attributes', []))}

        # 3. Canonicalize all defined functions
        canonical_functions = []
        for func in mod.functions:
            if func.is_declaration:
                continue
            
            canonical_func_str = canonicalize_function(func, global_attr_map)
            canonical_functions.append(canonical_func_str)
        
        # 4. Save the final canonical file
        canon_path = os.path.join(canonical_dir, f"{base_name}.canon.ll")
        with open(canon_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(canonical_functions))
        
        return (ll_file_path, "SUCCESS")
        
    except Exception as e:
        return (ll_file_path, f"ERROR: {e}")

# --- 2. STATEFUL CANONICALIZATION (AST WALK) ---

def canonicalize_function(func, global_attr_map):
    """
    Performs the stateful, two-pass canonicalization of a single
    llvmlite Function object, based on the spec.
    """
    
    # --- PASS 1: Build Mappings ---
    maps = {
        'reg': OrderedDict(),   # %5 -> DEF1
        'label': OrderedDict()  # %entry -> BLOCK_0
    }
    reg_counter = 0
    label_counter = 0

    # 1a. Map Arguments
    for arg in func.arguments:
        maps['reg'][f"%{arg.name}"] = f"ARG{reg_counter}"
        reg_counter += 1

    # 1b. Map Basic Blocks (Labels) and Instructions (Registers)
    for block in func.blocks:
        maps['label'][block.name] = f"BLOCK_{label_counter}"
        label_counter += 1
        
        for instr in block.instructions:
            if instr.name:
                maps['reg'][f"%{instr.name}"] = f"REG{reg_counter}"
                reg_counter += 1

    # --- PASS 2: Build Canonical String ---
    output = []
    
    # 2a. Build Function Header
    func_header = build_function_header(func, maps, global_attr_map)
    output.append(func_header)
    
    # 2b. Build Function Body
    for block in func.blocks:
        output.append(f"BLOCK_LABEL({maps['label'][block.name]})")
        
        for instr in block.instructions:
            output.append(f"  {canonicalize_instruction(instr, maps)}")

    # 2c. Build Function Footer
    output.append(f"FUNC_END @{func.name}")
    
    return "\n".join(output)


def build_function_header(func, maps, global_attr_map):
    """
    Builds the FUNC_START line from the spec.
    e.g.: FUNC_START @name (DEF0: TYPE_INT32) -> TYPE_INT32 [nounwind]
    """
    
    # Safely get return type
    try:
        if hasattr(func.type, 'element_type'):
            ret_type = map_type(func.type.element_type.return_type)
        elif hasattr(func.type, 'return_type'):
            ret_type = map_type(func.type.return_type)
        else:
            ret_type = "TYPE_UNKNOWN"
    except (AttributeError, RuntimeError):
        ret_type = "TYPE_UNKNOWN"
    
    arg_list = []
    for arg in func.arguments:
        arg_id = maps['reg'][f"%{arg.name}"]
        arg_type = map_type(arg.type)
        arg_attrs = " ".join([f"[{attr}]" for attr in arg.attributes])
        arg_list.append(f"{arg_id}: {arg_type} {arg_attrs}".strip())
    
    func_attrs_str = ""
    if func.attributes:
        try:
            # Try to get attribute reference index
            if hasattr(func.attributes, 'index'):
                attr_ref = f"#{func.attributes.index}"
                if attr_ref in global_attr_map:
                    attrs = re.findall(r'"(.*?)"|(\w+)', global_attr_map[attr_ref])
                    func_attrs_str = " ".join([f"[{a[1]}]" for a in attrs if a[1] and a[1] not in ["{", "}"]])
            else:
                # Direct attribute list - extract attribute names
                attr_list = []
                for attr in func.attributes:
                    attr_str = str(attr)
                    # Extract attribute names from the string representation
                    attr_names = re.findall(r'\b(\w+)\b', attr_str)
                    attr_list.extend([name for name in attr_names if name not in ['attributes', 'function']])
                if attr_list:
                    func_attrs_str = " ".join([f"[{attr}]" for attr in attr_list])
        except (AttributeError, RuntimeError, KeyError):
            pass

    return f"FUNC_START @{func.name} ({', '.join(arg_list)}) -> {ret_type} {func_attrs_str}".strip()


# --- 3. INSTRUCTION & TYPE MAPPERS ---

def canonicalize_instruction(instr, maps):
    """
    Translates a single llvmlite Instruction object into its
    canonical string representation.
    """
    
    lhs = ""
    if instr.name:
        lhs = f"{maps['reg'][f'%{instr.name}']} = "

    op = OPCODE_MAP.get(instr.opcode, f"OP_UNKNOWN({instr.opcode})")
    parts = [op]
    operands = []
    
    # Convert operands to list for indexing
    operands_list = list(instr.operands)
    
    # Special-case logic for complex instructions
    
    if op == 'OP_CALL':
        if len(operands_list) > 0:
            callee_operand = operands_list[-1]
            
            if hasattr(callee_operand, 'is_function') and callee_operand.is_function and callee_operand.name.startswith("llvm."):
                intrinsic_name = callee_operand.name.replace("llvm.", "").split('.')[0].upper().replace('-', '_')
                op = f"INTRINSIC_{intrinsic_name}"
                parts = [op]
                args = [canonicalize_operand(op, maps) for op in operands_list[:-1]]
                parts.append(f"({', '.join(args)}) -> {map_type(instr.type)}")
            else:
                args = [canonicalize_operand(op, maps) for op in operands_list]
                parts.append(f"({', '.join(args)}) -> {map_type(instr.type)}")

    elif op == 'OP_INVOKE':
        if len(operands_list) > 0:
            callee_operand = operands_list[-1]
            args = [canonicalize_operand(op, maps) for op in operands_list[:-1]]
            callee_str = canonicalize_operand(callee_operand, maps)
            
            if hasattr(instr, 'successors') and len(list(instr.successors)) >= 2:
                successors_list = list(instr.successors)
                to_label = canonicalize_operand(successors_list[0], maps, as_label=True)
                unwind_label = canonicalize_operand(successors_list[1], maps, as_label=True)
                
                parts.append(f"({callee_str}, {', '.join(args)}) -> {map_type(instr.type)}")
                parts.append(f"to {to_label} unwind {unwind_label}")
            else:
                parts.append(f"({callee_str}, {', '.join(args)}) -> {map_type(instr.type)}")
    
    elif op == 'OP_PHI':
        parts.append(map_type(instr.type))
        phi_pairs = []
        for i in range(0, len(operands_list), 2):
            if i + 1 < len(operands_list):
                val = canonicalize_operand(operands_list[i], maps)
                label_op = operands_list[i+1]
                label = canonicalize_operand(label_op, maps, as_label=True)
                phi_pairs.append(f"[ {val}, {label} ]")
        parts.append(", ".join(phi_pairs))

    elif op == 'OP_SWITCH':
        if len(operands_list) >= 2:
            val_op = canonicalize_operand(operands_list[0], maps)
            def_label = canonicalize_operand(operands_list[1], maps, as_label=True)
            parts.append(f"{val_op}, default {def_label}")
            
            jump_table = []
            for i in range(2, len(operands_list), 2):
                if i + 1 < len(operands_list):
                    case_val = canonicalize_operand(operands_list[i], maps)
                    case_label = canonicalize_operand(operands_list[i+1], maps, as_label=True)
                    jump_table.append(f"[ {case_val}, {case_label} ]")
            if jump_table:
                parts.append(" ".join(jump_table))

    elif op in ['OP_LANDINGPAD', 'OP_CATCHPAD', 'OP_CLEANUPPAD']:
        parts.append(map_type(instr.type))
        if hasattr(instr, 'is_cleanup') and instr.is_cleanup:
            parts.append('[cleanup]')
        if hasattr(instr, 'clauses'):
            clauses = []
            for clause in instr.clauses:
                clause_str = 'catch' if clause.is_catch else 'filter'
                clauses.append(f"{clause_str} {canonicalize_operand(clause.operand, maps)}")
            if clauses:
                parts.append(", ".join(clauses))

    elif op in ['OP_RESUME', 'OP_CATCHRET', 'OP_CLEANUPRET']:
        for op_item in operands_list:
            operands.append(canonicalize_operand(op_item, maps))
        parts.append(", ".join(operands))
    
    else:
        # Default handling for all other simple opcodes
        instr_str = str(instr)
        if instr.opcode in ['add', 'sub', 'mul', 'shl']:
            if 'nsw' in instr_str: parts.append('[nsw]')
            if 'nuw' in instr_str: parts.append('[nuw]')
        if instr.opcode == 'getelementptr' and 'inbounds' in instr_str:
            parts.append('[inbounds]')
        
        if instr.opcode in ['icmp', 'fcmp']:
            if hasattr(instr, 'predicate'):
                parts.append(f"[{instr.predicate}]")

        for op_item in operands_list:
            operands.append(canonicalize_operand(op_item, maps))
        parts.append(", ".join(operands))

    # Add instruction-level metadata
    if hasattr(instr, 'metadata'):
        metadata_parts = []
        for kind, md_node in instr.metadata.items():
            metadata_parts.append(f"!{kind} {str(md_node)}")
        if metadata_parts:
            parts.append(", ".join(metadata_parts))
    
    return f"{lhs}{' '.join(p for p in parts if p)}"


def canonicalize_operand(op, maps, as_label=False):
    """
    Translates a single llvmlite Value object into its
    canonical string representation (Sections 1 & 4).
    Enhanced error handling for type access.
    """
    
    op_str = str(op)
    
    # 4. Constant Mapping
    if hasattr(op, 'is_constant') and op.is_constant:
        if op_str == 'null': return "CONST_NULL"
        if op_str == 'undef': return "CONST_UNDEF"
        if op_str == 'zeroinitializer': return "CONST_ZERO"
        
        try:
            type_kind = op.type.kind
            if type_kind == llvm.TypeKind.INTEGER:
                if hasattr(op.type, 'width') and op.type.width == 1:
                    return f"CONST_BOOL({1 if 'true' in op_str else 0})"
                val = op.name if op.name else op_str.split()[-1]
                return f"CONST_INT({val})"
            
            if type_kind in [llvm.TypeKind.FLOAT, llvm.TypeKind.DOUBLE]:
                val = op.name if op.name else op_str.split()[-1]
                return f"CONST_FLOAT({val})"
            
            if hasattr(op, 'is_global_variable') and op.is_global_variable:
                return op.name
        except (AttributeError, RuntimeError):
            pass
        
        return f"CONST_UNKNOWN({op_str})"

    # 1. Identifier Mapping
    if hasattr(op, 'is_instruction') and op.is_instruction:
        return f"USE_{maps['reg'][f'%{op.name}']}"
    
    if hasattr(op, 'is_argument') and op.is_argument:
        return f"USE_{maps['reg'][f'%{op.name}']}"
    
    # Check if it's a basic block by checking if it's in our label map
    if hasattr(op, 'name') and op.name in maps['label']:
        prefix = "USE_" if as_label else ""
        return f"{prefix}{maps['label'][op.name]}"

    if hasattr(op, 'is_function') and op.is_function:
        return op.name
    
    if hasattr(op, 'is_global_variable') and op.is_global_variable:
        return op.name
    
    if hasattr(op, 'is_metadata') and op.is_metadata:
        return str(op)
    
    return f"UNKNOWN_OPERAND({op_str})"


def map_type(type_ref):
    """
    Recursively translates an llvmlite TypeRef object into its
    canonical string representation (Section 2).
    Handles opaque pointers and missing element_type attributes.
    """
    try:
        kind = type_ref.kind
    except (AttributeError, RuntimeError):
        return "TYPE_UNKNOWN"
    
    if kind == llvm.TypeKind.VOID: return "TYPE_VOID"
    if kind == llvm.TypeKind.INTEGER: 
        return f"TYPE_INT{type_ref.width if hasattr(type_ref, 'width') else '?'}"
    if kind == llvm.TypeKind.FLOAT: return "TYPE_FLOAT"
    if kind == llvm.TypeKind.DOUBLE: return "TYPE_DOUBLE"
    if kind == llvm.TypeKind.FP128: return "TYPE_FP128"
    if kind == llvm.TypeKind.PPC_FP128: return "TYPE_PPC_FP128"
    if kind == llvm.TypeKind.TOKEN: return "TYPE_TOKEN"
    if kind == llvm.TypeKind.METADATA: return "TYPE_METADATA"
    
    if kind == llvm.TypeKind.POINTER:
        # Handle opaque pointers (no element_type)
        if hasattr(type_ref, 'element_type'):
            try:
                element_type = map_type(type_ref.element_type)
                return f"TYPE_PTR({element_type})"
            except (AttributeError, RuntimeError):
                pass
        # Opaque pointer or error - just use generic pointer
        return "TYPE_PTR(TYPE_OPAQUE)"
    
    if kind == llvm.TypeKind.ARRAY:
        if hasattr(type_ref, 'element_type') and hasattr(type_ref, 'count'):
            try:
                element_type = map_type(type_ref.element_type)
                count = type_ref.count
                return f"TYPE_ARRAY({count}, {element_type})"
            except (AttributeError, RuntimeError):
                pass
        return "TYPE_ARRAY(?, TYPE_OPAQUE)"
    
    if kind == llvm.TypeKind.VECTOR:
        if hasattr(type_ref, 'element_type') and hasattr(type_ref, 'count'):
            try:
                element_type = map_type(type_ref.element_type)
                count = type_ref.count
                return f"TYPE_VECTOR({count}, {element_type})"
            except (AttributeError, RuntimeError):
                pass
        return "TYPE_VECTOR(?, TYPE_OPAQUE)"
    
    if kind == llvm.TypeKind.FUNCTION:
        try:
            ret_type = map_type(type_ref.return_type)
            param_types = [map_type(p) for p in type_ref.parameters]
            if hasattr(type_ref, 'is_var_arg') and type_ref.is_var_arg:
                param_types.append("...")
            return f"TYPE_FUNC({ret_type}, {', '.join(param_types)})"
        except (AttributeError, RuntimeError):
            return "TYPE_FUNC(TYPE_UNKNOWN)"
    
    if kind == llvm.TypeKind.STRUCT:
        try:
            if hasattr(type_ref, 'is_named') and type_ref.is_named:
                return f"TYPE_STRUCT_NAMED(@{type_ref.name})"
            else:
                if hasattr(type_ref, 'elements'):
                    elements = [map_type(e) for e in type_ref.elements]
                    return f"TYPE_STRUCT({', '.join(elements)})"
        except (AttributeError, RuntimeError):
            pass
        return "TYPE_STRUCT(TYPE_UNKNOWN)"
    
    return f"TYPE_UNKNOWN({kind})"


# --- 4. MAIN EXECUTION (Robust & Parallel) ---

def init_llvm_worker():
    """
    Initializer function for each worker process.
    This ensures LLVM is initialized *within* the process
    that will be using it.
    """
    # llvm.initialize() is deprecated and handled automatically in newer versions
    try:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
    except Exception:
        # If initialization fails or is already done, continue
        pass


def process_ir_file_wrapper(args):
    """Helper function to unpack arguments for executor.map"""
    return process_ir_file(*args)


if __name__ == "__main__":
    
    os.makedirs(CANONICAL_UNOPT_DIR, exist_ok=True)
    os.makedirs(CANONICAL_OPT_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    print("Starting Task 4: Robust Canonicalizing Tokenizer (AST-based, Parallel)...")
    
    # 1. Build the list of all tasks
    all_tasks = []
    unopt_files = glob.glob(os.path.join(FILTERED_UNOPT_DIR, "*_unopt.ll"))
    opt_files = glob.glob(os.path.join(FILTERED_OPT_DIR, "*_opt.ll"))
    
    for f_path in unopt_files:
        base = os.path.basename(f_path).replace(".ll", "")
        all_tasks.append((f_path, base, CANONICAL_UNOPT_DIR, METADATA_DIR))
    
    for f_path in opt_files:
        base = os.path.basename(f_path).replace(".ll", "")
        all_tasks.append((f_path, base, CANONICAL_OPT_DIR, METADATA_DIR))

    total = len(all_tasks)
    print(f"Found {total} total files to tokenize...")
    
    error_count = 0
    processed_count = 0
    
    # 2. Run tasks in parallel and show real-time results
    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_llvm_worker
    ) as executor:
        
        futures = [executor.submit(process_ir_file_wrapper, task) for task in all_tasks]
        
        for future in as_completed(futures):
            (f_path, status) = future.result()
            
            if status == "SUCCESS":
                # Print a dot for success to avoid clutter, and flush stdout
                print(".", end="")
                sys.stdout.flush()
            elif status == "SKIPPED_EMPTY":
                # Print an 'S' for skipped
                print("S", end="")
                sys.stdout.flush()
            else:
                # Print the full error message
                print(f"\nFailed to process {f_path}: {status}")
                error_count += 1
            
            processed_count += 1
            if processed_count % 100 == 0: # Newline every 100 files
                print(f" [{processed_count}/{total}]")

    print(f"\n\nTask 4 complete.")
    if error_count > 0:
        print(f"WARNING: {error_count} files failed to process. See logs above.")
    print(f"Final dataset is ready in 'dataset_final/'")