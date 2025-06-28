#!/usr/bin/env fish
# Quick activation script for Fish shell

if test -d .venv
    source .venv/bin/activate.fish
    echo "✅ Virtual environment activated!"
    echo "Python: "(which python)
else
    echo "❌ No virtual environment found. Run 'make setup' first."
end