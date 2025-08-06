#!/bin/bash
# Switch Transformer Evaluation Runner
# Complete 5×5 experiment matrix with analysis

echo "🚀 Switch Transformer Prefetching Evaluation"
echo "=============================================="
echo "Strategies: A(OnDemand), B(Oracle), C(MultiLook), D(TopK), E(Intelligent)"
echo "Batch sizes: 1, 2, 4, 8, 16"
echo "Total experiments: 25"
echo "Expected time: 2-4 hours"
echo ""

# Create results directory
mkdir -p results

# Change to scripts directory
cd scripts

echo "📊 Step 1: Running all experiments..."
python run_batch_evaluation.py \
    --runs 10 \
    --parallel \
    --max-parallel 2 \
    --output-dir ../results

echo ""
echo "📊 Step 2: Analyzing results..."
python analyze_results.py \
    --input ../results \
    --plots

echo ""
echo "📊 Step 3: Generating CSV/Excel reports..."
python generate_csv_report.py \
    --input ../results \
    --format both

echo ""
echo "✅ Evaluation completed!"
echo "📁 Results available in: results/"
echo ""
echo "📄 Key files:"
echo "   - results/switch_results.xlsx (Excel report)"
echo "   - results/switch_results.csv (Main results)" 
echo "   - results/plots/switch_prefetching_analysis.png (Performance plots)"
echo "   - results/README.md (Summary and usage)"
echo ""
echo "🎯 Next: Review the Excel file for detailed analysis and recommendations!"