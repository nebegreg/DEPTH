#!/bin/bash

# Depth Anything v3 GUI Launcher Script
# ======================================
# Script de lancement simplifié pour l'application GUI

set -e  # Exit on error

echo "========================================="
echo "  Depth Anything v3 - GUI Professional  "
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python 3 found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠${NC} Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment found"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import PyQt6" 2>/dev/null; then
    echo -e "${YELLOW}⚠${NC} Dependencies not installed"
    echo "Installing dependencies... (this may take a few minutes)"

    # Install PyTorch
    echo "Installing PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

    # Install other dependencies
    echo "Installing GUI dependencies..."
    pip install -r requirements_gui.txt --quiet

    # Install Depth Anything v3
    if [ -d "Depth-Anything-3-main" ]; then
        echo "Installing Depth Anything v3..."
        cd Depth-Anything-3-main
        pip install -e . --quiet
        cd ..
    else
        echo -e "${RED}Error: Depth-Anything-3-main directory not found${NC}"
        echo "Please extract Depth-Anything-3-main.zip first"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} All dependencies installed"
else
    echo -e "${GREEN}✓${NC} Dependencies already installed"
fi

# Check CUDA availability
echo ""
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print('${GREEN}✓ CUDA available: ' + torch.cuda.get_device_name(0) + '${NC}')
    print('  VRAM: ' + str(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)) + ' GB')
else:
    print('${YELLOW}⚠ CUDA not available - running on CPU (slower)${NC}')
"

echo ""
echo "========================================="
echo "Launching Depth Anything v3 GUI..."
echo "========================================="
echo ""

# Launch the application
python depth_anything_gui.py

# Deactivate virtual environment on exit
deactivate
