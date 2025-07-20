#!/bin/bash

# Define variables
DATA_PATH="/mnt/c/Internship/Contrastive Learning/Datasets/PACS"
BATCH_SIZE=64
EPOCHS=100
ARCH="resnet50"
OUT_DIM=128

# Create directories for experiment results
mkdir -p experiment_results
mkdir -p evaluation_results

# Run each experiment
run_experiment() {
    SOURCE_DOMAINS="$1"
    TARGET_DOMAIN="$2"
    EXP_NAME="$3"

    echo "===== Running experiment: $EXP_NAME ====="

    # Train SimCLR on source domains
    python run.py -data "$DATA_PATH" -dataset-name pacs --source-domains $SOURCE_DOMAINS --target-domain $TARGET_DOMAIN --experiment-name $EXP_NAME --arch $ARCH --batch-size $BATCH_SIZE --epochs $EPOCHS --out_dim $OUT_DIM --fp16-precision

    if [ $? -ne 0 ]; then
        echo "Error during training for $EXP_NAME"
        exit 1
    fi

    # Create config file if it doesn't exist
    CONFIG_FILE="experiment_results/$EXP_NAME/config.yml"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Creating config file for $EXP_NAME"
        mkdir -p "experiment_results/$EXP_NAME"
        echo "arch: $ARCH" > "$CONFIG_FILE"
        echo "out_dim: $OUT_DIM" >> "$CONFIG_FILE"
    fi

    # Evaluate on target domain
    python eval_pacs.py -data "$DATA_PATH" --target-domain "$TARGET_DOMAIN" --checkpoint "experiment_results/$EXP_NAME/model.pth.tar" --arch "$ARCH" --batch-size "$BATCH_SIZE"
}

# Running all experiments
run_experiment "Art Cartoon Sketch" "Photo" "Art_Cartoon_Sketch-to-Photo"
run_experiment "Photo Cartoon Sketch" "Art" "Photo_Cartoon_Sketch-to-Art"
run_experiment "Photo Art Sketch" "Cartoon" "Photo_Art_Sketch-to-Cartoon"
run_experiment "Photo Art Cartoon" "Sketch" "Photo_Art_Cartoon-to-Sketch"

echo "All experiments completed!"
