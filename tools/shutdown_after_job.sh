#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage:
  tools/shutdown_after_job.sh [options] -- command [args...]
  tools/shutdown_after_job.sh [options] --pid PID

Options:
  --pid PID                 Watch an existing process instead of running a new command.
  --delay-minutes MINUTES   Delay before shutdown after the job exits. Default: 1.
  --log PATH                Append wrapper events to this log file.
  -h, --help                Show this help.

Environment:
  SHUTDOWN_DELAY_MINUTES    Default delay if --delay-minutes is not provided.
  CHECK_INTERVAL_SECONDS    Poll interval for --pid mode. Default: 60.
  SHUTDOWN_DRY_RUN=1        Log the shutdown command without executing it.
EOF
}

delay_minutes="${SHUTDOWN_DELAY_MINUTES:-1}"
check_interval="${CHECK_INTERVAL_SECONDS:-60}"
watch_pid=""
log_path=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pid)
      if [ "$#" -lt 2 ]; then
        echo "--pid requires a value" >&2
        exit 2
      fi
      watch_pid="$2"
      shift 2
      ;;
    --delay-minutes)
      if [ "$#" -lt 2 ]; then
        echo "--delay-minutes requires a value" >&2
        exit 2
      fi
      delay_minutes="$2"
      shift 2
      ;;
    --log)
      if [ "$#" -lt 2 ]; then
        echo "--log requires a value" >&2
        exit 2
      fi
      log_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [ -n "$log_path" ]; then
  mkdir -p "$(dirname "$log_path")"
fi

log_msg() {
  local msg
  msg="$(date '+%Y-%m-%d %H:%M:%S') $*"
  echo "$msg"
  if [ -n "$log_path" ]; then
    echo "$msg" >> "$log_path"
  fi
}

schedule_shutdown() {
  log_msg "Scheduling shutdown in ${delay_minutes} minute(s)."
  sync
  if [ "${SHUTDOWN_DRY_RUN:-0}" = "1" ]; then
    log_msg "DRY RUN: shutdown -h +${delay_minutes}"
    return 0
  fi
  /usr/bin/shutdown -h "+${delay_minutes}"
}

if [ -n "$watch_pid" ]; then
  if ! kill -0 "$watch_pid" 2>/dev/null; then
    log_msg "PID ${watch_pid} is not running; scheduling shutdown now."
    schedule_shutdown
    exit 0
  fi
  log_msg "Watching PID ${watch_pid}; poll interval ${check_interval}s."
  while kill -0 "$watch_pid" 2>/dev/null; do
    sleep "$check_interval"
  done
  log_msg "PID ${watch_pid} exited."
  schedule_shutdown
  exit 0
fi

if [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi

log_msg "Starting command: $*"
"$@"
status=$?
log_msg "Command exited with status ${status}."
schedule_shutdown
exit "$status"
