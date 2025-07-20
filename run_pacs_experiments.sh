#!/bin/bash

# Define variables
DATA_PATH="/kaggle/input/pacs-dataset/PACS"  # Adjust path for Windows
BATCH_SIZE=128
EPOCHS=100
ARCH="resnet50"
OUT_DIM=128

# Create directory for experiment results
mkdir -p experiment_results

# Function to run an experiment
run_experiment() {
    SOURCE_DOMAINS=$1
    TARGET_DOMAIN=$2
    EXP_NAME="${SOURCE_DOMAINS// /_}-to-${TARGET_DOMAIN}"
    
    echo "===== Running experiment: $EXP_NAME ====="
    
    # Train SimCLR on source domains
    python run.py \
        -data "$DATA_PATH" \
        --dataset-name pacs \
        --source-domains $SOURCE_DOMAINS \
        --target-domain $TARGET_DOMAIN \
        --experiment-name $EXP_NAME \
        --arch $ARCH \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --out_dim $OUT_DIM \
        --fp16-precision
    
    # Evaluate on target domain
    python eval_pacs.py \
        -data "$DATA_PATH" \
        --target-domain $TARGET_DOMAIN \
        --checkpoint "experiment_results/$EXP_NAME/model.pth.tar" \
        --arch $ARCH \
        --batch-size $BATCH_SIZE
}

# Run all domain combinations (leave-one-domain-out)
run_experiment "Art Cartoon Sketch" "Photo"
run_experiment "Photo Cartoon Sketch" "Art"
run_experiment "Photo Art Sketch" "Cartoon"
run_experiment "Photo Art Cartoon" "Sketch"

# Print summary of results
echo "===== Experiment Results Summary ====="
for FILE in evaluation_results/*.txt; do
    echo "Results from $FILE:"
    cat "$FILE"
    echo "------------------------"
done