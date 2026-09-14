#!/bin/bash
# entrypoint.sh

# Initialize micromamba shell
eval "$(micromamba shell hook --shell bash)"
micromamba activate occenv

# Symlink setup (if needed)
[ -e ./psr ] || ln -s /brepdgt/psr ./psr
[ -e ./blender ] || ln -s /brepdgt/blender ./blender

# Start interactive shell
exec bash
