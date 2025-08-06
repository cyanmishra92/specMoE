#!/bin/bash
# SpecMoE Complete Evaluation Suite
# Runs comprehensive evaluation across all architectures and strategies

set -e  # Exit on error

echo "🚀 SpecMoE Complete Evaluation Suite"
echo "===================================="

# Configuration
OUTPUT_DIR="results/complete_evaluation_$(date +%Y%m%d_%H%M%S)"
NUM_RUNS=10
BATCH_SIZES="1,8,16,32"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/evaluation_suite.log"
echo "📝 Logging to: $LOG_FILE"

# Function to run evaluation with error handling
run_evaluation() {
    local arch=$1
    local description=$2
    local extra_args=${3:-""}
    
    echo ""
    echo "🔄 Running $description..."
    echo "Architecture: $arch"
    echo "Extra args: $extra_args"
    
    if python -m src.evaluation.run_evaluation \
        --architecture "$arch" \
        --batch_sizes "$BATCH_SIZES" \
        --num_runs "$NUM_RUNS" \
        --output_dir "$OUTPUT_DIR/$arch" \
        --verbose \
        $extra_args >> "$LOG_FILE" 2>&1; then
        echo "✅ $description completed successfully"
    else
        echo "❌ $description failed (check log for details)"
        return 1
    fi
}

# Start evaluation suite
echo "Starting evaluation suite at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Number of runs per configuration: $NUM_RUNS"
echo "Batch sizes: $BATCH_SIZES"

# Initialize log
echo "SpecMoE Complete Evaluation Suite" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "=====================================" >> "$LOG_FILE"

# 1. Switch Transformer Evaluation
echo ""
echo "1️⃣ Switch Transformer Evaluation"
echo "--------------------------------"
run_evaluation "switch_transformer" "Switch Transformer Base Evaluation" \
    "--strategy intelligent,topk,multilook,oracle --cache_size 50"

# 2. Qwen MoE Evaluation  
echo ""
echo "2️⃣ Qwen MoE Evaluation"
echo "----------------------"
run_evaluation "qwen_moe" "Qwen MoE Base Evaluation" \
    "--strategy intelligent,topk,multilook,oracle --cache_size 160"

# 3. Comparative Evaluation
echo ""
echo "3️⃣ Comparative Analysis"
echo "----------------------"
run_evaluation "comparative" "Comparative Analysis with Baselines" \
    "--include_baselines --enable_deduplication"

# 4. Expert Deduplication Analysis
echo ""
echo "4️⃣ Expert Deduplication Study"
echo "-----------------------------"
run_evaluation "deduplication" "Expert Deduplication Analysis" \
    "--batch_sizes 1,2,4,8,16,32,64"

# 5. Cross-Architecture Comparison
echo ""
echo "5️⃣ Cross-Architecture Analysis"
echo "------------------------------"
echo "🔄 Running cross-architecture comparison..."

if python scripts/evaluation/cross_architecture_comparison.py \
    --output_dir "$OUTPUT_DIR/cross_architecture" \
    --batch_sizes "$BATCH_SIZES" \
    --num_runs "$NUM_RUNS" >> "$LOG_FILE" 2>&1; then
    echo "✅ Cross-architecture comparison completed"
else
    echo "❌ Cross-architecture comparison failed"
fi

# 6. Hardware-Specific Evaluations
echo ""
echo "6️⃣ Hardware-Specific Analysis"
echo "-----------------------------"

hardware_configs=("rtx_4090" "a100_80gb" "h100_80gb")
for hw in "${hardware_configs[@]}"; do
    echo "🔄 Running evaluation for $hw..."
    if python -m src.evaluation.run_evaluation \
        --architecture switch_transformer \
        --strategy intelligent \
        --batch_sizes "$BATCH_SIZES" \
        --hardware "$hw" \
        --num_runs 5 \
        --output_dir "$OUTPUT_DIR/hardware_$hw" \
        --verbose >> "$LOG_FILE" 2>&1; then
        echo "✅ $hw evaluation completed"
    else
        echo "❌ $hw evaluation failed"
    fi
done

# 7. Scaling Analysis
echo ""
echo "7️⃣ Scaling Analysis"
echo "-------------------"
echo "🔄 Running scaling analysis..."

if python scripts/evaluation/scaling_analysis.py \
    --architectures "switch_transformer,qwen_moe" \
    --strategies "intelligent,topk" \
    --batch_sizes "1,2,4,8,16,32,64,128" \
    --cache_sizes "25,50,100,200" \
    --output_dir "$OUTPUT_DIR/scaling" >> "$LOG_FILE" 2>&1; then
    echo "✅ Scaling analysis completed"
else
    echo "❌ Scaling analysis failed"
fi

# 8. Generate Comprehensive Report
echo ""
echo "8️⃣ Generating Comprehensive Report"
echo "-----------------------------------"
echo "🔄 Compiling results and generating report..."

if python scripts/evaluation/generate_comprehensive_report.py \
    --input_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_DIR/COMPREHENSIVE_EVALUATION_REPORT.md" \
    --include_plots \
    --include_statistics >> "$LOG_FILE" 2>&1; then
    echo "✅ Comprehensive report generated"
else
    echo "❌ Report generation failed"
fi

# Final summary
echo ""
echo "======================================"
echo "🎉 Evaluation Suite Completed!"
echo "======================================"
echo ""
echo "📊 Results Summary:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log file: $LOG_FILE"
echo "  Duration: $(date)"
echo ""

# Check if all major components completed
if [[ -f "$OUTPUT_DIR/switch_transformer/evaluation_summary.json" && \
      -f "$OUTPUT_DIR/qwen_moe/evaluation_summary.json" && \
      -f "$OUTPUT_DIR/comparative/comparative_results.json" ]]; then
    echo "✅ All major evaluations completed successfully"
    echo ""
    echo "📚 Next steps:"
    echo "  • View comprehensive report: cat $OUTPUT_DIR/COMPREHENSIVE_EVALUATION_REPORT.md"
    echo "  • Analyze results: python scripts/analysis/analyze_evaluation_results.py --input_dir $OUTPUT_DIR"
    echo "  • Generate publication plots: python scripts/visualization/create_publication_plots.py --input_dir $OUTPUT_DIR"
    echo ""
    exit 0
else
    echo "⚠️  Some evaluations may have failed. Check the log file for details."
    echo "📝 Log file: $LOG_FILE"
    echo ""
    exit 1
fi