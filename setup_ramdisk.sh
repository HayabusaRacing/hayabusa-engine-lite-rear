#!/bin/bash

# Create RAM disk for CFD cases
# Size: 20GB for 5 parallel cases (4GB per case should be plenty)
RAMDISK_SIZE="20G"
RAMDISK_PATH="/tmp/ramdisk"

echo "Setting up RAM disk for CFD acceleration..."

# Create mount point
sudo mkdir -p $RAMDISK_PATH

# Mount RAM disk
sudo mount -t tmpfs -o size=$RAMDISK_SIZE tmpfs $RAMDISK_PATH

# Make it writable and owned by current user
sudo chmod 777 $RAMDISK_PATH
sudo chown $USER:$USER $RAMDISK_PATH

# Create subdirectories (AFTER ownership change)
mkdir -p $RAMDISK_PATH/case
mkdir -p $RAMDISK_PATH/tmp

# Ensure subdirectories have correct ownership too
sudo chown -R $USER:$USER $RAMDISK_PATH

echo "RAM disk mounted at $RAMDISK_PATH with $RAMDISK_SIZE"
echo "Available space:"
df -h $RAMDISK_PATH

echo ""
echo "To unmount when done, run: sudo umount $RAMDISK_PATH"
