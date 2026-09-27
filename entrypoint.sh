#!/bin/sh
# Fix volume mount ownership before dropping privileges.
# Railway (and most container runtimes) create volume mount points as root:root 0755.
# This script runs as root, chowns writable paths to the aurora user, then
# re-execs the process as aurora so the runtime is still non-root.
exec gosu aurora "$@"
