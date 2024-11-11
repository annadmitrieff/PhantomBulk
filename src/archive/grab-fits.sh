#!/bin/bash

# Set the destination directory
destination="fitsfiles"

# Loop through each RT.fits file in the sim directories
for filepath in sim_*/data_1300/RT.fits; do
    # Extract the simulation number (e.g., sim_0099 -> 99)
    sim_num=$(echo "$filepath" | sed -E 's|sim_0*([0-9]+)/.*|\1|')
    
    # Set the new filename based on the simulation number
    new_filename="${sim_num}.fits"
    
    # Rename the file in place
    mv "$filepath" "$(dirname "$filepath")/$new_filename"
    
    # Copy the renamed file to the destination directory
    cp "$(dirname "$filepath")/$new_filename" "$destination/"
done

echo "All files have been renamed and copied to the fitsfiles directory."

