#!/bin/bash

# Define variables
DATA_PATH="/kaggle/input/pacs-dataset/PACS"
BATCH_SIZE=64
EPOCHS=100
ARCH="resnet50"
OUT_DIM=128
MEMORY_SIZE=4096
PROTOTYPE_WEIGHT=0.5

# Create directories for experiment results
mkdir -p experiment_results
mkdir -p evaluation_results

# First, let's check if the PACS dataset exists and has the right structure
echo "Checking dataset structure..."
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Dataset directory $DATA_PATH does not exist!"
    exit 1
fi

# List available directories to help with debugging
echo "Available directories in $DATA_PATH:"
ls -la "$DATA_PATH"

# Add a function to find domain directories recursively
find_domain_dirs() {
    local base_dir=$1
    local domain=$2
    
    # Try direct path first
    if [ -d "$base_dir/$domain" ]; then
        echo "$base_dir/$domain"
        return 0
    fi
    
    # Search recursively
    local result=$(find "$base_dir" -type d -name "$domain" | head -n 1)
    if [ -n "$result" ]; then
        echo "$result"
        return 0
    fi
    
    return 1
}

# Check for each domain
domains=("Photo" "Art" "Cartoon" "Sketch")
found_domains=()

for domain in "${domains[@]}"; do
    domain_dir=$(find_domain_dirs "$DATA_PATH" "$domain")
    if [ -n "$domain_dir" ]; then
        echo "Found $domain domain at: $domain_dir"
        found_domains+=("$domain")
        
        # Check if it has class subdirectories
        class_count=$(find "$domain_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "  - Contains $class_count class directories"
    else
        echo "WARNING: $domain domain not found!"
    fi
done

if [ ${#found_domains[@]} -eq 0 ]; then
    echo "Error: No domain directories found. Please check your dataset structure."
    exit 1
fi

echo "Found ${#found_domains[@]} out of 4 domains. Continuing with experiments..."

# Run each experiment
run_experiment() {
    SOURCE_DOMAINS="$1"
    TARGET_DOMAIN="$2"
    EXP_NAME="ProtoDACL_${SOURCE_DOMAINS// /_}-to-${TARGET_DOMAIN}"

    echo "===== Running experiment: $EXP_NAME ====="
    
    # Check if target domain exists
    target_dir=$(find_domain_dirs "$DATA_PATH" "$TARGET_DOMAIN")
    if [ -z "$target_dir" ]; then
        echo "Error: Target domain $TARGET_DOMAIN not found. Skipping experiment."
        return 1
    fi
    
    # Check if source domains exist
    missing_sources=()
    for source in $SOURCE_DOMAINS; do
        source_dir=$(find_domain_dirs "$DATA_PATH" "$source")
        if [ -z "$source_dir" ]; then
            missing_sources+=("$source")
        fi
    done
    
    if [ ${#missing_sources[@]} -gt 0 ]; then
        echo "Error: The following source domains are missing: ${missing_sources[*]}"
        echo "Skipping experiment."
        return 1
    fi

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
        echo "Error during training for $EXP_NAME (status: $?)"
        echo "Skipping evaluation and continuing with next experiment"
        return 1
    fi

    # Evaluate on target domain
    python eval_protodacl.py -data "$DATA_PATH" \
                          --target-domain "$TARGET_DOMAIN" \
                          --checkpoint "experiment_results/$EXP_NAME/model.pth.tar" \
                          --arch "$ARCH" \
                          --batch-size "$BATCH_SIZE"
                          
    if [ $? -ne 0 ]; then
        echo "Error during evaluation for $EXP_NAME (status: $?)"
        return 1
    fi
    
    return 0
}

# Running all experiments
failed_experiments=()

run_experiment "Art Cartoon Sketch" "Photo" || failed_experiments+=("Art_Cartoon_Sketch-to-Photo")
run_experiment "Photo Cartoon Sketch" "Art" || failed_experiments+=("Photo_Cartoon_Sketch-to-Art")
run_experiment "Photo Art Sketch" "Cartoon" || failed_experiments+=("Photo_Art_Sketch-to-Cartoon")
run_experiment "Photo Art Cartoon" "Sketch" || failed_experiments+=("Photo_Art_Cartoon-to-Sketch")

echo "All experiments completed!"

if [ ${#failed_experiments[@]} -gt 0 ]; then
    echo "The following experiments had errors:"
    for exp in "${failed_experiments[@]}"; do
        echo "  - $exp"
    done
    exit 1
else
    echo "All experiments completed successfully!"
    exit 0
fi