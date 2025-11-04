# LLVM Compiler Optimization Data Pipeline

## 🎯 Project Overview

This project implements a complete data pipeline designed to test whether Large Language Models (LLMs) can effectively learn compiler optimization patterns. The core innovation is the creation of a novel "canonical" text format for LLVM IR that removes semantic noise while preserving optimization-relevant structural information.

### Research Hypothesis
By training LLMs on canonicalized LLVM IR data (free from arbitrary register names, boilerplate metadata, and other semantic noise), models can learn actual optimization patterns more effectively than with raw IR representations.

### Key Innovation
- **Canonical LLVM IR Format**: A structured representation that normalizes:
  - Register names (`%5` → `DEF_REG1`)
  - Basic block labels (`label1` → `BLOCK_LABEL_0`) 
  - Instruction opcodes (`add` → `OP_ADD`)
  - Type representations (`i32` → `TYPE_INT32`)
  - Metadata separation for cleaner training data

## 📁 Project Structure

```
CD_Project/
├── README.md                    # This file
├── .gitignore                   # Git ignore patterns
├── platform.info               # System information
│
├── generate_corpus.sh           # Task 1: C code generation using csmith
├── generate_ir.py              # Task 2: LLVM IR compilation (-O0 and -O2)
├── analyze_corpus.py           # Task 3: Quality filtering and analysis
├── tokeniser.py                # Task 4: Canonicalization pipeline
│
├── dataset/                    # Raw generated data (gitignored)
│   ├── raw_c/                  # Original C files from csmith
│   ├── unoptimized_ir/         # -O0 LLVM IR files
│   └── optimized_ir/           # -O2 LLVM IR files
│
├── dataset_filtered/           # Quality-filtered data
│   ├── manifest.json           # Filtering analysis and statistics
│   ├── unoptimized_ir/         # High-quality unoptimized IR pairs
│   └── optimized_ir/           # High-quality optimized IR pairs
│
└── dataset_final/              # Canonicalized final dataset
    ├── metadata/               # Separated metadata (.meta.json files)
    ├── unoptimized_ir/         # Canonical unoptimized IR (.canon.ll)
    └── optimized_ir/           # Canonical optimized IR (.canon.ll)
```

## 🚀 Pipeline Overview

### Task 1: C Code Generation (`generate_corpus.sh`)
- **Tool**: Csmith random C program generator
- **Output**: 10,000 syntactically valid C files
- **Quality Control**: 
  - 10-second timeout per file
  - Clang syntax validation
  - Error handling and resilience

### Task 2: IR Generation (`generate_ir.py`)
- **Input**: Raw C files from Task 1
- **Process**: Parallel compilation with Clang
- **Output**: 
  - Unoptimized IR files (`*_unopt.ll`) using `-O0`
  - Optimized IR files (`*_opt.ll`) using `-O2`
- **Features**: 
  - Multi-threaded processing (8 workers)
  - Timeout protection (30s per compilation)
  - Csmith header inclusion (`-I/usr/include/csmith`)

### Task 3: Quality Filtering (`analyze_corpus.py`)
- **Input**: Raw IR pairs from Task 2
- **Analysis Metrics**:
  - Instruction count analysis
  - Basic block count analysis
  - Opcode distribution analysis
- **Filtering Criteria**:
  - Minimum complexity (≥10 instructions, ≥2 basic blocks)
  - Optimization impact (instruction count changes)
  - Parsing validity
- **Output**: 
  - 7,663 high-quality IR pairs (76.63% retention rate)
  - Detailed manifest with statistics and discard reasons

### Task 4: Canonicalization (`tokeniser.py`)
- **Input**: Filtered IR pairs from Task 3
- **Process**: AST-based transformation using llvmlite
- **Key Transformations**:
  - Register normalization (`%tmp.1` → `DEF_REG1`)
  - Label standardization (`bb1` → `BLOCK_LABEL_0`)
  - Opcode mapping (`add` → `OP_ADD`)
  - Type canonicalization (`i32` → `TYPE_INT32`)
  - Metadata separation and storage
- **Output**: 
  - 15,326 canonical IR files
  - Separated metadata files for training flexibility
  - Error-resistant processing with detailed logging

## 📊 Dataset Statistics

| Metric | Count | Notes |
|--------|--------|-------|
| **Generated C Files** | 10,000 | Raw csmith output |
| **Raw IR Pairs** | 10,000 | Unopt + Opt pairs |
| **Filtered IR Pairs** | 7,663 | 76.63% retention |
| **Final Canonical Files** | 15,326 | Ready for ML training |

### Quality Filtering Results
- **Kept**: 7,663 pairs (76.63%)
- **Discarded Reasons**:
  - Parse failures/empty files
  - Trivial programs (too simple)
  - No optimization impact
  - Missing pair files

## 🛠️ Usage Instructions

### Prerequisites
```bash
# Install required tools
sudo apt-get install csmith clang llvm-dev
pip install llvmlite

# Clone repository
git clone <repository-url>
cd CD_Project
```

### Full Pipeline Execution
```bash
# 1. Generate C corpus (adjust TARGET_COUNT in script)
./generate_corpus.sh

# 2. Compile to LLVM IR
python generate_ir.py

# 3. Filter and analyze quality
python analyze_corpus.py

# 4. Canonicalize for ML training
python tokeniser.py
```

### Individual Script Usage

#### Generate C Files
```bash
# Edit generate_corpus.sh to set:
TARGET_COUNT=10000  # Number of files to generate
START_NUM=0         # Starting file number

./generate_corpus.sh
```

#### Generate IR Files
```bash
# Processes all .c files in dataset/raw_c/
# Outputs to dataset/unoptimized_ir/ and dataset/optimized_ir/
python generate_ir.py
```

#### Filter Dataset
```bash
# Analyzes and filters IR pairs
# Creates manifest.json with detailed statistics
python analyze_corpus.py
```

#### Canonicalize IR
```bash
# Transforms filtered IR to canonical format
# Separates metadata and normalizes structure
python tokeniser.py
```

## 📋 Canonical Format Specification

### Register Mapping
```llvm
; Original IR
%tmp.1 = add i32 %a, %b
%result = mul i32 %tmp.1, 2

; Canonical Format
DEF_REG1 = OP_ADD TYPE_INT32 DEF_ARG0, DEF_ARG1
DEF_REG2 = OP_MUL TYPE_INT32 DEF_REG1, CONST_INT(2)
```

### Block Labels
```llvm
; Original IR
entry:
  br label %loop.body
loop.body:
  br label %exit
exit:
  ret i32 0

; Canonical Format
BLOCK_LABEL_0:
  OP_BR BLOCK_LABEL_1
BLOCK_LABEL_1:
  OP_BR BLOCK_LABEL_2
BLOCK_LABEL_2:
  OP_RET TYPE_INT32 CONST_INT(0)
```

### Function Structure
```llvm
; Canonical function format
FUNC_START @function_name (DEF_ARG0: TYPE_INT32) -> TYPE_INT32 [nounwind]
BLOCK_LABEL_0:
  DEF_REG1 = OP_ADD TYPE_INT32 DEF_ARG0, CONST_INT(1)
  OP_RET TYPE_INT32 DEF_REG1
FUNC_END @function_name
```

## 🔧 Configuration Options

### Generate Corpus Settings
```bash
# In generate_corpus.sh
TARGET_COUNT=10000    # Total files to generate
START_NUM=0          # Starting file number
```

### IR Generation Settings
```python
# In generate_ir.py
MAX_WORKERS = 8      # Parallel compilation threads
timeout=30           # Compilation timeout (seconds)
```

### Filtering Thresholds
```python
# In analyze_corpus.py
MIN_INSTRUCTIONS = 10    # Minimum instruction count
MIN_BLOCKS = 2          # Minimum basic block count
```

## 📈 Performance Metrics

### Processing Times (Approximate)
- **C Generation**: ~2-3 hours for 10k files
- **IR Compilation**: ~1-2 hours (8 workers)
- **Quality Filtering**: ~15-30 minutes
- **Canonicalization**: ~45-60 minutes

### Resource Requirements
- **Disk Space**: ~5-10GB for complete dataset
- **Memory**: 8GB+ recommended for parallel processing
- **CPU**: Multi-core recommended for optimal performance

## 🐛 Troubleshooting

### Common Issues

#### Csmith Not Found
```bash
sudo apt-get install csmith
# or build from source if package unavailable
```

#### LLVM/Clang Version Issues
```bash
# Install specific LLVM version
sudo apt-get install llvm-14-dev clang-14
```

#### Python Dependencies
```bash
pip install llvmlite
# If installation fails, try:
conda install llvmlite
```

#### Permission Issues
```bash
chmod +x generate_corpus.sh
```

### Error Recovery
- **Partial failures**: Scripts are designed to skip failed files and continue
- **Interrupted runs**: Most scripts can resume from where they left off
- **Memory issues**: Reduce `MAX_WORKERS` in parallel processing scripts

## 📝 Output Files

### Manifest Structure (`dataset_filtered/manifest.json`)
```json
{
  "kept_files": {
    "file_id": {
      "unopt_stats": {
        "instr_count": 45,
        "block_count": 8,
        "top_opcodes": {"OP_ADD": 12, "OP_LOAD": 8}
      },
      "opt_stats": {
        "instr_count": 32,
        "block_count": 6,
        "top_opcodes": {"OP_ADD": 8, "OP_LOAD": 6}
      }
    }
  },
  "discarded_files": {
    "file_id": "REASON_FOR_DISCARD"
  }
}
```

### Metadata Structure (`dataset_final/metadata/*.meta.json`)
```json
{
  "source_filename": "example.c",
  "target_triple": "x86_64-unknown-linux-gnu",
  "datalayout": "e-m:e-i64:64-f80:128-n8:16:32:64-S128",
  "globals": ["@global_var = ..."],
  "declarations": ["declare i32 @printf(...)"],
  "attributes": ["#0 = { nounwind }"],
  "module_flags": ["!0 = !{...}"],
  "named_metadata": ["!llvm.ident = !{...}"]
}
```

## 🎓 Research Applications

### Machine Learning Training
- **Input**: Canonical unoptimized IR
- **Target**: Canonical optimized IR
- **Benefits**: 
  - Reduced vocabulary size
  - Consistent structure
  - Focus on optimization patterns

### Analysis Opportunities
- Optimization pattern frequency analysis
- Instruction reduction statistics
- Basic block transformation patterns
- Opcode distribution changes

## 📚 Technical Details

### LLVM IR Canonicalization
The canonicalization process uses llvmlite's AST parsing to:
1. Parse LLVM IR into structured representations
2. Build symbol tables for registers and labels
3. Apply systematic renaming rules
4. Separate metadata from core logic
5. Generate clean, training-ready text

### Quality Metrics
- **Instruction Count**: Proxy for program complexity
- **Basic Block Count**: Control flow complexity indicator
- **Optimization Delta**: Instruction count reduction measure
- **Parse Success**: Syntactic validity check

## 🔬 Future Extensions

### Potential Improvements
- **Advanced Filtering**: Cyclomatic complexity analysis
- **Multi-Optimization Levels**: -O1, -O3, -Os variants
- **Target Architecture**: Multiple target platforms
- **Language Variants**: C++, Rust, other LLVM frontends

### Research Directions
- **LLM Training**: Seq2seq models for optimization prediction
- **Pattern Analysis**: Statistical analysis of optimization patterns
- **Benchmark Creation**: Standard evaluation datasets
- **Tool Integration**: IDE plugins for optimization suggestions

---

**Note**: This pipeline is not completed fully.
