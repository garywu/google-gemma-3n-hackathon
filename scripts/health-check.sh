#!/bin/bash
# Project Health Check

echo "🏥 Project Health Check"
echo "====================="

# Git status
echo "📝 Git Status:"
if [[  -n $(git status -s)  ]]; then
    echo "⚠️  Uncommitted changes found"
else
    echo "✅ Working directory clean"
fi

# More checks can be added based on project type
echo ""
echo "Health check complete!"
