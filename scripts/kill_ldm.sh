#!/bin/bash

SUFFIX="_ldm"
PIDFILE="/home/yb107/logs/train${SUFFIX}.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "❌ PID file not found: $PIDFILE"
    exit 1
fi

PID=$(cat $PIDFILE)
echo "Killing validation process group with PGID = $PID ..."
kill -9 -$PID
rm -f "$PIDFILE"
echo "✅ Validation process and subprocesses killed."