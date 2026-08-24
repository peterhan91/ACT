#!/bin/bash
#SBATCH --job-name=Download_RSNA2023
#SBATCH --output=rsna2023_%j.out
#SBATCH --time=21-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL

# Load environment
source ~/.bashrc

# Set destination directory
DEST_DIR="/path/to/data/RSNA2023/"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Change to destination directory
cd "$DEST_DIR"

# Log start time
echo "Starting RSNA 2023 Abdominal Trauma Detection download at $(date)"

# Download the competition data
echo "Downloading RSNA 2023 Abdominal Trauma Detection dataset..."
kaggle competitions download -c rsna-2023-abdominal-trauma-detection

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully at $(date)"
    echo "Data saved to: $DEST_DIR"
    
    # Unzip the downloaded files
    echo "Extracting downloaded files..."
    for zip_file in *.zip; do
        if [ -f "$zip_file" ]; then
            echo "Extracting $zip_file..."
            unzip -q "$zip_file"
            echo "Removing $zip_file..."
            rm "$zip_file"
        fi
    done
    
    # Display directory size and file count
    echo "Directory size:"
    du -sh "$DEST_DIR"
    echo "Number of files:"
    find "$DEST_DIR" -type f | wc -l
    
    # List contents
    echo "Contents of $DEST_DIR:"
    ls -la "$DEST_DIR"
    
else
    echo "Download failed at $(date)"
    echo "Make sure you have:"
    echo "1. Accepted the competition rules at https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection"
    echo "2. Valid kaggle.json API token at ~/.kaggle/kaggle.json"
    exit 1
fi

echo "RSNA 2023 download job finished at $(date)"