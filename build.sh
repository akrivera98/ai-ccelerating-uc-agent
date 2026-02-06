#!/bin/bash
echo "Building submission template"
python build_submission.py --results $1

echo "Creating submission"
python create_submission.py

echo "Done!"