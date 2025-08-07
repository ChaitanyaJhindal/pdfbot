#!/bin/bash
# Build script for Render deployment
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt
echo "Build completed successfully!"
