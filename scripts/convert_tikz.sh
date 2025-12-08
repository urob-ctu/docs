#!/bin/bash
# filepath: /home/michal/Documents/Skola/ucitel/UROB/urob-docs/scripts/convert_tikz.sh

# Create temporary cache directory
CACHE_DIR=$(mktemp -d)
echo "Using temporary directory: $CACHE_DIR"

cd assets/tex
for file in *.tex; do
    echo "Processing $file..."
    
    # Compile to temp directory
    pdflatex -output-directory="$CACHE_DIR" "$file"
    
    # Convert PDF to SVG
    pdf2svg "$CACHE_DIR/${file%.tex}.pdf" "../images/${file%.tex}.svg"
    
    echo "Generated ../images/${file%.tex}.svg"
done

# Clean up temporary files
rm -rf "$CACHE_DIR"
echo "Done! Temporary files cleaned up."
