#!/bin/bash
#SBATCH --job-name=Download_STOIC2021
#SBATCH --output=stoic2021_%j.out
#SBATCH --time=21-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL

# Load environment
source ~/.bashrc

# Set destination directory
DEST_DIR="/path/to/data/STOIC2021/"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Log start time
echo "Starting STOIC2021 download at $(date)"

# Download the public training set from AWS S3
echo "Downloading STOIC2021 public training data..."
aws s3 cp s3://stoic2021-training/ "$DEST_DIR" --recursive --no-sign-request

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully at $(date)"
    echo "Data saved to: $DEST_DIR"
    
    # Display directory size and file count
    echo "Directory size:"
    du -sh "$DEST_DIR"
    echo "Number of files:"
    find "$DEST_DIR" -type f | wc -l
else
    echo "Download failed at $(date)"
    exit 1
fi

echo "STOIC2021 download job finished at $(date)"