#!/bin/bash
# Script to completely clean up the old directory structure
# This script removes all files and directories from the old structure

# Confirm before proceeding
echo "This script will completely remove the old directory structure."
echo "Make sure you have verified that the new structure is working correctly."
read -p "Are you sure you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 1
fi

# Remove the entire core directory
echo "Removing the entire core directory..."
rm -rf nexusml/core

echo "Cleanup complete!"