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

## 🚧 Future Work

### Phase 1: Detokenization Pipeline (`detokenizer.py`)

The next critical component is building a detokenizer that can convert the canonical format back to valid LLVM IR. This enables end-to-end evaluation of LLM-generated optimizations.

#### Key Requirements:
- **Reverse Mapping**: Convert canonical tokens back to LLVM IR syntax
- **Metadata Reconstruction**: Reintegrate separated metadata from `.meta.json` files
- **Validation**: Ensure generated IR is syntactically and semantically valid
- **Error Handling**: Graceful handling of malformed LLM outputs

#### Implementation Strategy:
```python
# Proposed detokenizer structure
class LLVMDetokenizer:
    def __init__(self, metadata_path):
        self.metadata = load_metadata(metadata_path)
        self.reverse_maps = build_reverse_mappings()
    
    def detokenize(self, canonical_ir):
        """Convert canonical IR back to valid LLVM IR"""
        # 1. Parse canonical tokens
        # 2. Rebuild symbol tables
        # 3. Generate LLVM IR syntax
        # 4. Reintegrate metadata
        # 5. Validate output
        pass
```

#### Validation Framework:
- **Syntax Check**: `llvm-as` compilation test
- **Semantic Check**: Type consistency verification
- **Execution Test**: Compare runtime behavior with original
- **Optimization Verification**: Measure actual performance improvements

### Phase 2: LLM Training and Fine-tuning

#### Dataset Preparation
```python
# Training data structure
{
    "input": "FUNC_START @func (DEF_ARG0: TYPE_INT32) -> TYPE_INT32...",
    "target": "FUNC_START @func (DEF_ARG0: TYPE_INT32) -> TYPE_INT32...",
    "metadata": {...}
}
```

#### Model Architecture Options

##### Option 1: Transformer Encoder-Decoder
```python
# Sequence-to-sequence model for IR transformation
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Custom tokenizer for canonical IR
class CanonicalIRTokenizer:
    def __init__(self):
        self.vocab = build_canonical_vocab()
        self.special_tokens = ["<START>", "<END>", "<FUNC>", "<BLOCK>"]
```

##### Option 2: Code-Specific Models
- **CodeT5**: Pre-trained on code understanding tasks
- **CodeBERT**: Understanding of programming semantics
- **InCoder**: Infilling capabilities for optimization tasks

#### Training Configuration
```yaml
# training_config.yaml
model:
  type: "t5-base"
  max_length: 2048
  vocab_size: 50000

training:
  batch_size: 16
  learning_rate: 5e-5
  epochs: 10
  warmup_steps: 1000
  gradient_accumulation: 4

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  max_examples: 15326
```

#### Evaluation Metrics
- **BLEU Score**: Token-level similarity with ground truth
- **Code Similarity**: AST-based structural comparison
- **Compilation Success Rate**: Percentage of valid outputs
- **Optimization Effectiveness**: Instruction count reduction
- **Performance Benchmarks**: Runtime improvement measurement

### Phase 3: Advanced Training Strategies

#### Curriculum Learning
```python
# Progressive training strategy
training_stages = [
    {"complexity": "simple", "epochs": 3, "criteria": "block_count < 5"},
    {"complexity": "medium", "epochs": 4, "criteria": "5 <= block_count < 15"},
    {"complexity": "complex", "epochs": 3, "criteria": "block_count >= 15"}
]
```

#### Multi-Task Learning
- **Primary Task**: Unoptimized → Optimized IR transformation
- **Auxiliary Tasks**:
  - Opcode prediction
  - Block count estimation
  - Optimization type classification (-O1, -O2, -O3)

#### Reinforcement Learning Integration
```python
# Reward function based on actual performance
def optimization_reward(original_ir, optimized_ir):
    return {
        "instruction_reduction": calculate_instruction_delta(original_ir, optimized_ir),
        "compilation_success": verify_compilation(optimized_ir),
        "execution_correctness": verify_semantics(original_ir, optimized_ir),
        "performance_gain": measure_runtime_improvement(original_ir, optimized_ir)
    }
```

### Phase 4: Evaluation and Benchmarking

#### Benchmark Suite Development
```python
# Comprehensive evaluation framework
class OptimizationBenchmark:
    def __init__(self):
        self.test_cases = load_test_suite()
        self.baseline_optimizers = ["gcc -O2", "clang -O2", "llvm opt"]
    
    def evaluate_model(self, model, test_set):
        results = {
            "compilation_rate": 0,
            "optimization_effectiveness": 0,
            "semantic_correctness": 0,
            "performance_comparison": {}
        }
        return results
```

#### Performance Comparison
- **Against Traditional Optimizers**: LLVM opt, GCC, Clang
- **Instruction Count Reduction**: Quantitative improvement measure
- **Runtime Performance**: Actual execution speed improvements
- **Memory Usage**: Optimization impact on memory patterns

### Phase 5: Production Integration

#### Tool Development
```bash
# Command-line optimization tool
llm-opt input.ll --model path/to/trained/model --output optimized.ll
```

#### IDE Integration
- **VS Code Extension**: Real-time optimization suggestions
- **Language Server**: Integration with existing development workflows
- **Batch Processing**: Large codebase optimization capabilities

#### Deployment Options
- **Local Models**: On-device optimization for privacy
- **Cloud API**: Scalable optimization service
- **Edge Computing**: Mobile/embedded optimization

### Implementation Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1** | 3-4 weeks | Detokenizer + validation framework |
| **Phase 2** | 6-8 weeks | Trained LLM models + evaluation |
| **Phase 3** | 4-6 weeks | Advanced training strategies |
| **Phase 4** | 3-4 weeks | Comprehensive benchmarking |
| **Phase 5** | 4-6 weeks | Production tools + integration |

### Research Questions to Address

1. **Model Architecture**: Which transformer architecture works best for IR optimization?
2. **Training Data**: What's the optimal balance of complexity in training examples?
3. **Evaluation Metrics**: How do we measure "optimization quality" beyond instruction count?
4. **Generalization**: Can models trained on csmith data optimize real-world code?
5. **Interpretability**: Can we understand what optimization patterns the model learns?

### Expected Outcomes

#### Academic Contributions
- **Novel Dataset**: First large-scale canonicalized LLVM IR dataset
- **Methodology**: Reproducible pipeline for compiler optimization research
- **Empirical Results**: Quantitative analysis of LLM optimization capabilities
- **Benchmarks**: Standard evaluation suite for future research

#### Practical Applications
- **Developer Tools**: AI-assisted optimization suggestions
- **Compiler Enhancement**: ML-guided optimization passes
- **Education**: Teaching optimization concepts through AI explanations
- **Research Platform**: Foundation for future compiler research

---

**Note**: This represents the complete roadmap from data pipeline to production-ready LLM-based compiler optimization tools.
