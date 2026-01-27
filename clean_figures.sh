#!/bin/bash

# Script to recursively remove all directories starting with "figures"
# but preserve those named exactly "figures_ref"

echo "Searching for 'figures*' directories to remove (excluding 'figures_ref')..."
echo "=========================================="

# Find all directories starting with "figures" but exclude "figures_ref"
# Use -print0 and read -d $'\0' to handle directories with spaces
found_dirs=()
while IFS= read -r -d $'\0' dir; do
    # Get the basename of the directory
    basename=$(basename "$dir")

    # Skip if the directory is named exactly "figures_ref"
    if [ "$basename" = "figures_ref" ]; then
        echo "Preserving: $dir"
        continue
    fi

    # Add to list of directories to remove
    found_dirs+=("$dir")
done < <(find . -type d -name "figures*" -print0)

# Check if we found any directories to remove
if [ ${#found_dirs[@]} -eq 0 ]; then
    echo "No directories to remove."
    exit 0
fi

# Display directories that will be removed
echo ""
echo "The following directories will be removed:"
echo "------------------------------------------"
for dir in "${found_dirs[@]}"; do
    echo "  $dir"
done
echo "------------------------------------------"
echo "Total: ${#found_dirs[@]} director(y/ies)"
echo ""

# Ask for confirmation
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Remove the directories
echo ""
echo "Removing directories..."
for dir in "${found_dirs[@]}"; do
    rm -rf "$dir"
    if [ $? -eq 0 ]; then
        echo "  ✓ Removed: $dir"
    else
        echo "  ✗ Failed to remove: $dir"
    fi
done

echo ""
echo "Done!"
