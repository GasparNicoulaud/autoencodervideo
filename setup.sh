#!/bin/bash

echo "Setting up Video Autoencoder Environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw data/processed
mkdir -p models/pretrained models/checkpoints
mkdir -p output/reconstructions output/samples output/interpolations
mkdir -p logs

echo "Setup complete! Activate the environment with: source venv/bin/activate"