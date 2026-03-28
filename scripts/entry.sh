#!/bin/bash
set -e

# ---- Fix permissions for CyVerse VICE ----
# CyVerse VICE may run the container with a UID different from the built-in
# 'sat' user (UID 1000). Ensure the runtime user has a home directory and
# writable Jupyter runtime directories regardless of which UID is active.

RUNTIME_UID=$(id -u)
RUNTIME_GID=$(id -g)

# If running as a UID without a passwd entry (common in k8s/VICE), create one
if ! whoami &>/dev/null 2>&1; then
  echo "sat:x:${RUNTIME_UID}:${RUNTIME_GID}:SAT User:/home/sat:/bin/bash" >> /etc/passwd 2>/dev/null || true
fi

# Ensure home directory exists and is writable
export HOME="${HOME:-/home/sat}"
mkdir -p "$HOME" 2>/dev/null || true
# If we can't write to $HOME, fall back to /tmp as home
if ! touch "$HOME/.test_write" 2>/dev/null; then
  export HOME="/tmp/sat_home"
  mkdir -p "$HOME"
fi
rm -f "$HOME/.test_write" 2>/dev/null || true

# Ensure Jupyter runtime directories are writable
mkdir -p "$HOME/.jupyter" \
         "$HOME/.local/share/jupyter/runtime" \
         "$HOME/.local/share/jupyter/kernels" \
         "$HOME/.ipython" \
         2>/dev/null || true

# Fix ownership of workdir if running as root (e.g., when VICE overrides USER)
if [ "$RUNTIME_UID" = "0" ]; then
  # Running as root — make things accessible to all users
  chmod -R a+rwX /opt/segmentanytree 2>/dev/null || true
  chmod -R a+rwX /data 2>/dev/null || true
  chmod -R a+rwX /tmp/sat_cache 2>/dev/null || true
fi

# ---- Configure iRODS environment for CyVerse Data Store access ----
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
  cp "/data-store/iplant/home/$IPLANT_USER/.gitconfig" "$HOME/"
fi

# Copy user's SSH keys if available from data store
if [ -n "$IPLANT_USER" ] && [ -d "/data-store/iplant/home/$IPLANT_USER/.ssh" ]; then
  cp -r "/data-store/iplant/home/$IPLANT_USER/.ssh" "$HOME/"
  chmod 700 "$HOME/.ssh"
  chmod 600 "$HOME/.ssh"/* 2>/dev/null || true
fi

# Start JupyterLab
exec jupyter lab \
  --no-browser \
  --ServerApp.token="" \
  --ServerApp.password="" \
  --ServerApp.allow_remote_access=true \
  --ServerApp.allow_origin="*" \
  --ServerApp.allow_unauthenticated_access=true \
  --ServerApp.disable_check_xsrf=true \
  --ServerApp.terminado_settings='{"shell_command": ["/bin/bash"]}' \
  --ip="0.0.0.0" \
  --port=8888 \
  --notebook-dir="${NOTEBOOK_DIR:-$HOME}"
