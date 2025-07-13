#!/bin/bash

RAMDISK_PATH="/tmp/ramdisk"

echo "Unmounting RAM disk at $RAMDISK_PATH..."

# Check if mounted
if mount | grep -q "$RAMDISK_PATH"; then
    sudo umount $RAMDISK_PATH
    echo "RAM disk unmounted successfully"
    
    # Remove directory
    sudo rmdir $RAMDISK_PATH 2>/dev/null || echo "Directory not empty, leaving it"
else
    echo "RAM disk not mounted at $RAMDISK_PATH"
fi

echo "RAM disk cleanup completed"
