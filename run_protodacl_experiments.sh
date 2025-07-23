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

    if [ $? -ne 0 ]; then
        echo "Error during training for $EXP_NAME"
        exit 1
    fi

    # Evaluate on target domain
    python eval_protodacl.py -data "$DATA_PATH" \
                          --target-domain "$TARGET_DOMAIN" \
                          --checkpoint "experiment_results/$EXP_NAME/model.pth.tar" \
                          --arch "$ARCH" \
                          --batch-size "$BATCH_SIZE"
}

# Running all experiments
run_experiment "Art Cartoon Sketch" "Photo" 
run_experiment "Photo Cartoon Sketch" "Art"
run_experiment "Photo Art Sketch" "Cartoon"
run_experiment "Photo Art Cartoon" "Sketch"

echo "All experiments completed!"