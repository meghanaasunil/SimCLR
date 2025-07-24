#!/bin/bash

# Define variables
DATA_PATH="./datasets/PACS"
BATCH_SIZE=64
EPOCHS=100
ARCH="resnet50"
OUT_DIM=128
MEMORY_SIZE=4096
PROTOTYPE_WEIGHT=0.5

# Create directories for experiment results
mkdir -p experiment_results
mkdir -p evaluation_results

# Run each experiment
run_experiment() {
    SOURCE_DOMAINS="$1"
    TARGET_DOMAIN="$2"
    EXP_NAME="ProtoDACL_${SOURCE_DOMAINS// /_}-to-${TARGET_DOMAIN}"

    echo "===== Running experiment: $EXP_NAME ====="

    # Train ProtoDACL on source domains
    python run_protodacl.py -data "$DATA_PATH" \
                         --dataset-name pacs \
                         --source-domains $SOURCE_DOMAINS \
                         --target-domain $TARGET_DOMAIN \
                         --experiment-name $EXP_NAME \
                         --arch $ARCH \
                         --batch-size $BATCH_SIZE \
                         --epochs $EPOCHS \
                         --out_dim $OUT_DIM \
                         --memory-size $MEMORY_SIZE \
                         --prototype-weight $PROTOTYPE_WEIGHT \
                         --fp16-precision

    TRAIN_STATUS=$?
    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "Error during training for $EXP_NAME (status: $TRAIN_STATUS)"
        echo "Skipping evaluation and continuing with next experiment"
        return $TRAIN_STATUS
    fi

    # Check if model exists before evaluation
    if [ ! -f "experiment_results/$EXP_NAME/model.pth.tar" ]; then
        echo "Error: Model file not found at experiment_results/$EXP_NAME/model.pth.tar"
        echo "Skipping evaluation and continuing with next experiment"
        return 1
    fi

    # Evaluate on target domain
    python eval_protodacl.py -data "$DATA_PATH" \
                          --target-domain "$TARGET_DOMAIN" \
                          --checkpoint "experiment_results/$EXP_NAME/model.pth.tar" \
                          --arch "$ARCH" \
                          --batch-size "$BATCH_SIZE"

    EVAL_STATUS=$?
    if [ $EVAL_STATUS -ne 0 ]; then
        echo "Error during evaluation for $EXP_NAME (status: $EVAL_STATUS)"
        return $EVAL_STATUS
    fi
    
    echo "Experiment $EXP_NAME completed successfully!"
    return 0
}

# Running all experiments with error tracking
FAILED_EXPERIMENTS=()

run_experiment "Art Cartoon Sketch" "Photo"
if [ $? -ne 0 ]; then
    FAILED_EXPERIMENTS+=("Art_Cartoon_Sketch-to-Photo")
fi

run_experiment "Photo Cartoon Sketch" "Art"
if [ $? -ne 0 ]; then
    FAILED_EXPERIMENTS+=("Photo_Cartoon_Sketch-to-Art")
fi

run_experiment "Photo Art Sketch" "Cartoon"
if [ $? -ne 0 ]; then
    FAILED_EXPERIMENTS+=("Photo_Art_Sketch-to-Cartoon")
fi

run_experiment "Photo Art Cartoon" "Sketch"
if [ $? -ne 0 ]; then
    FAILED_EXPERIMENTS+=("Photo_Art_Cartoon-to-Sketch")
fi

echo "All experiments completed!"

# Report any failed experiments
if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo "The following experiments had errors:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
    exit 1
else
    echo "All experiments completed successfully!"
    exit 0
fi