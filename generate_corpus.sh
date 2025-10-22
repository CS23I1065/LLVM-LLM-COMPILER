#!/bin/bash

# --- SETTINGS ---
# You can change these two variables

# 1. The total number of files you want to create
TARGET_COUNT=10000

# 2. The number you want to start naming your files from
START_NUM=0
# -----------------


# --- SCRIPT ---
OUTPUT_DIR="dataset/raw_c"
GENERATED_COUNT=0
CURRENT_FILE_NUM=$START_NUM

echo "Starting C file generation..."

# This loop will run TARGET_COUNT times
while [ $GENERATED_COUNT -lt $TARGET_COUNT ]; do
    TEMP_FILE=$(mktemp)
    
    # The file name is now just the number
    FILE_NAME="$OUTPUT_DIR/$CURRENT_FILE_NUM.c"

    # 1. Run csmith with a timeout (resilience)
    timeout 10s csmith > $TEMP_FILE

    if [ $? -eq 0 ]; then
        # 2. Add a simple validation check (resilience)
        clang -fsyntax-only $TEMP_FILE &> /dev/null

        if [ $? -eq 0 ]; then
            # Success! Move the file and increment counters.
            mv $TEMP_FILE $FILE_NAME
            echo "Generated valid file: $FILE_NAME"
            
            GENERATED_COUNT=$((GENERATED_COUNT + 1))
            CURRENT_FILE_NUM=$((CURRENT_FILE_NUM + 1))
        else
            # Failed compilation, discard
            rm $TEMP_FILE
            echo "Discarding file (compile error)."
        fi
    else
        # Failed csmith run (timeout or other error), discard
        rm $TEMP_FILE
        echo "Discarding file (csmith error or timeout)."
    fi
done

echo "Corpus generation complete. Total files: $GENERATED_COUNT"
