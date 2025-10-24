#!/bin/bash
#
# Generate animated GIF of US election efficiency gaps over time
# Creates SVG files for each congressional election year from 2000-2024
# Then combines them into an animated GIF using ImageMagick
#

set -e  # Exit on any error

# Activate virtual environment
source .venv/bin/activate

# Create temporary directory for individual year SVGs
mkdir -p temp_animation

echo "Generating SVG files for each election year..."

# Generate SVG for each congressional election year (even years from 1980-2024)
years=(1980 1982 1984 1986 1988 1990 1992 1994 1996 1998 2000 2002 2004 2006 2008 2010 2012 2014 2016 2018 2020 2022 2024)
for year in "${years[@]}"; do
    echo "Processing year $year..."
    python render-graph3.py us-states.pickle "temp_animation/us-states-$year.svg" "$year"
done

echo "Converting SVGs to animated GIF..."

# Use ImageMagick to create animated GIF
# -delay 10 = 10 centiseconds = 100ms per frame for regular frames
# -delay 200 = 200 centiseconds = 2000ms = 2 seconds for decade markers
# -loop 0 = infinite loop

# Create animation with pauses every 10 years (1980, 1990, 2000, 2010, 2020) plus final 2024
convert \
  -delay 200 temp_animation/us-states-1980.svg \
  -delay 10 temp_animation/us-states-{1982,1984,1986,1988}.svg \
  -delay 200 temp_animation/us-states-1990.svg \
  -delay 10 temp_animation/us-states-{1992,1994,1996,1998}.svg \
  -delay 200 temp_animation/us-states-2000.svg \
  -delay 10 temp_animation/us-states-{2002,2004,2006,2008}.svg \
  -delay 200 temp_animation/us-states-2010.svg \
  -delay 10 temp_animation/us-states-{2012,2014,2016,2018}.svg \
  -delay 200 temp_animation/us-states-2020.svg \
  -delay 10 temp_animation/us-states-2022.svg \
  -delay 200 temp_animation/us-states-2024.svg \
  -loop 0 us-election-animation.gif

echo "Cleaning up temporary files..."
rm -rf temp_animation

echo "Animation created: us-election-animation.gif"
echo "Total frames: 23 (years 1980-2024, every 2 years)"