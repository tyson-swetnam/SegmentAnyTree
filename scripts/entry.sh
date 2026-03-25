#!/bin/bash
set -e

# Configure iRODS environment for CyVerse Data Store access
mkdir -p "$HOME/.irods"
cat > "$HOME/.irods/irods_environment.json" << EOF
{
  "irods_host": "data.cyverse.org",
  "irods_port": 1247,
  "irods_user_name": "${IPLANT_USER:-anonymous}",
  "irods_zone_name": "iplant"
}
EOF

# Copy user's git config if available from data store
if [ -n "$IPLANT_USER" ] && [ -f "/data-store/iplant/home/$IPLANT_USER/.gitconfig" ]; then
  cp "/data-store/iplant/home/$IPLANT_USER/.gitconfig" ~/
fi

# Copy user's SSH keys if available from data store
if [ -n "$IPLANT_USER" ] && [ -d "/data-store/iplant/home/$IPLANT_USER/.ssh" ]; then
  cp -r "/data-store/iplant/home/$IPLANT_USER/.ssh" ~/
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/* 2>/dev/null || true
fi

# Start JupyterLab
exec jupyter lab --no-browser --LabApp.token="" --LabApp.password="" --ip="0.0.0.0" --port=8888 --notebook-dir=/opt/segmentanytree
