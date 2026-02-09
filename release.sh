#!/usr/bin/env bash
# Release script for Options Spread Conviction Engine
# Usage: ./release.sh <version> "changelog message"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION=${1:-}
CHANGELOG=${2:-"Release v$VERSION"}

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version> [changelog message]"
    echo "Example: ./release.sh 1.0.1 'Fixed documentation'"
    exit 1
fi

echo "🚀 Releasing Options Spread Conviction Engine v$VERSION"
echo ""

# Update version in _meta.json
echo "📦 Updating _meta.json..."
sed -i "s/\"version\": \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/\"version\": \"$VERSION\"/" _meta.json

# Stage changes
echo "📋 Staging changes..."
git add .

# Commit
echo "💾 Creating commit..."
git commit -m "v$VERSION - $CHANGELOG" || echo "Nothing to commit"

# Tag
echo "🏷️  Creating Git tag..."
git tag -a "v$VERSION" -m "$CHANGELOG" || {
    echo "Tag v$VERSION already exists. Delete it with: git tag -d v$VERSION"
    exit 1
}

# Push to GitHub
echo "☁️  Pushing to GitHub..."
git push origin main
git push origin "v$VERSION"

echo ""
echo "✅ GitHub release complete!"
echo ""

# Publish to ClawHub
echo "📤 Publishing to ClawHub..."
clawhub publish . \
  --slug options-spread-conviction-engine \
  --name "Options Spread Conviction Engine" \
  --version "$VERSION" \
  --changelog "$CHANGELOG"

echo ""
echo "✅ Release v$VERSION complete!"
echo ""
echo "GitHub: https://github.com/AdamNaghs/Options-Spread-Conviction-Engine/releases/tag/v$VERSION"
echo "ClawHub: Updated options-spread-conviction-engine to v$VERSION"
echo ""
echo "Users can update with:"
echo "  clawhub update options-spread-conviction-engine"
