#!/bin/bash
# Shared library for tt-lang remote tools
# Source this after loading remote.conf.

# Run a command on the remote with the full environment.
# Usage: remote_run <command> [args...]
remote_run() {
    if [ -n "$REMOTE_HOST" ] && [ -n "$REMOTE_CONTAINER" ]; then
        local DOCKER_CMD="docker"
        [ "${DOCKER_SUDO:-0}" = "1" ] && DOCKER_CMD="sudo docker"
        ssh "$REMOTE_HOST" "$DOCKER_CMD exec -i $REMOTE_CONTAINER bash -c 'source ~/.bashrc 2>/dev/null; $*'"
    else
        $REMOTE_SHELL "$@"
    fi
}

# Copy a local file to the remote environment
# Usage: remote_copy_file <local_path> <remote_dest_path>
remote_copy_file() {
    local LOCAL_PATH="$1"
    local REMOTE_DEST="$2"

    if [ -n "$REMOTE_HOST" ] && [ -n "$REMOTE_CONTAINER" ]; then
        local DOCKER_CMD="docker"
        [ "${DOCKER_SUDO:-0}" = "1" ] && DOCKER_CMD="sudo docker"

        local TEMP_REMOTE="/tmp/ttl_copy_$$_$(basename "$LOCAL_PATH")"
        scp -q "$LOCAL_PATH" "$REMOTE_HOST:$TEMP_REMOTE" && \
        ssh "$REMOTE_HOST" "$DOCKER_CMD cp '$TEMP_REMOTE' '$REMOTE_CONTAINER:$REMOTE_DEST'"
    else
        cat "$LOCAL_PATH" | $REMOTE_SHELL bash -l -c "cat > '$REMOTE_DEST'"
    fi
}

# Copy a file from the remote environment to local
# Usage: remote_copy_from <remote_path> <local_path>
remote_copy_from() {
    local REMOTE_PATH="$1"
    local LOCAL_PATH="$2"

    if [ -n "$REMOTE_HOST" ] && [ -n "$REMOTE_CONTAINER" ]; then
        local DOCKER_CMD="docker"
        [ "${DOCKER_SUDO:-0}" = "1" ] && DOCKER_CMD="sudo docker"

        local TEMP_REMOTE="/tmp/ttl_copy_$$_$(basename "$REMOTE_PATH")"
        ssh "$REMOTE_HOST" "$DOCKER_CMD cp '$REMOTE_CONTAINER:$REMOTE_PATH' '$TEMP_REMOTE'" && \
        scp -q "$REMOTE_HOST:$TEMP_REMOTE" "$LOCAL_PATH"
    else
        remote_run cat "$REMOTE_PATH" > "$LOCAL_PATH"
    fi
}
