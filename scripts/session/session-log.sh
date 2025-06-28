#!/usr/bin/env bash
# Session Log Script - Log activities to current session
# Part of claude-init enhancement for persistent context tracking

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="${PWD}"
SESSION_DIR="${PROJECT_ROOT}/.claude"
SESSION_FILE="${SESSION_DIR}/session.json"

# Utility functions
log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Log activity to session
log_activity() {
  local message="$1"
  local activity_type="${2:-general}"
  local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  if [[ ! -f $SESSION_FILE ]]; then
    log_error "No active session found"
    echo "Start a new session with: make session-start"
    exit 1
  fi

  # Create activity entry
  local activity=$(
    cat <<EOF
{
    "timestamp": "$timestamp",
    "type": "$activity_type",
    "description": "$message"
}
EOF
  )

  # Update session file with new activity
  local temp_file="${SESSION_FILE}.tmp"
  jq --argjson activity "$activity" '.activities += [$activity]' "$SESSION_FILE" >"$temp_file"
  mv "$temp_file" "$SESSION_FILE"

  # Also append to session log file
  local session_id=$(jq -r '.id' "$SESSION_FILE")
  local log_file="${SESSION_DIR}/history/${session_id}.log"
  echo "[$timestamp] $activity_type: $message" >>"$log_file"

  log_success "Activity logged to session"
}

# Auto-detect activity type from message
detect_activity_type() {
  local message="$1"
  local lower_message=$(echo "$message" | tr '[:upper:]' '[:lower:]')

  # Check for common patterns
  if [[ $lower_message =~ (fix|bug|error|issue) ]]; then
    echo "bug-fix"
  elif [[ $lower_message =~ (feat|feature|add|implement) ]]; then
    echo "feature"
  elif [[ $lower_message =~ (test|spec) ]]; then
    echo "testing"
  elif [[ $lower_message =~ (doc|documentation|readme) ]]; then
    echo "documentation"
  elif [[ $lower_message =~ (refactor|clean|optimize) ]]; then
    echo "refactoring"
  elif [[ $lower_message =~ (deploy|release|publish) ]]; then
    echo "deployment"
  elif [[ $lower_message =~ (review|pr|pull.request) ]]; then
    echo "code-review"
  elif [[ $lower_message =~ (meeting|discuss|plan) ]]; then
    echo "planning"
  else
    echo "general"
  fi
}

# Log file modification
log_file_modification() {
  local file="$1"

  if [[ ! -f $SESSION_FILE ]]; then
    return
  fi

  # Check if file is already in the list
  local exists=$(jq --arg file "$file" '.files_modified | index($file)' "$SESSION_FILE")

  if [[ $exists == "null" ]]; then
    # Add file to modified list
    local temp_file="${SESSION_FILE}.tmp"
    jq --arg file "$file" '.files_modified += [$file]' "$SESSION_FILE" >"$temp_file"
    mv "$temp_file" "$SESSION_FILE"
  fi
}

# Log command execution
log_command() {
  local command="$1"
  local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  if [[ ! -f $SESSION_FILE ]]; then
    return
  fi

  # Create command entry
  local cmd_entry=$(
    cat <<EOF
{
    "timestamp": "$timestamp",
    "command": "$command"
}
EOF
  )

  # Update session file
  local temp_file="${SESSION_FILE}.tmp"
  jq --argjson cmd "$cmd_entry" '.commands_executed += [$cmd]' "$SESSION_FILE" >"$temp_file"
  mv "$temp_file" "$SESSION_FILE"
}

# Log GitHub issue work
log_issue_work() {
  local issue_number="$1"
  local issue_title="${2:-}"
  local action="${3:-worked}"

  if [[ ! -f $SESSION_FILE ]]; then
    return
  fi

  # Create issue entry
  local issue_entry=$(
    cat <<EOF
{
    "number": $issue_number,
    "title": "$issue_title",
    "status": "$action"
}
EOF
  )

  # Check if issue already in list
  local exists=$(jq --arg num "$issue_number" '.issues_worked[] | select(.number == ($num | tonumber))' "$SESSION_FILE" 2>/dev/null)

  if [[ -z $exists ]]; then
    # Add new issue
    local temp_file="${SESSION_FILE}.tmp"
    jq --argjson issue "$issue_entry" '.issues_worked += [$issue]' "$SESSION_FILE" >"$temp_file"
    mv "$temp_file" "$SESSION_FILE"
  else
    # Update existing issue status
    local temp_file="${SESSION_FILE}.tmp"
    jq --arg num "$issue_number" --arg status "$action" \
      '(.issues_worked[] | select(.number == ($num | tonumber)) | .status) = $status' \
      "$SESSION_FILE" >"$temp_file"
    mv "$temp_file" "$SESSION_FILE"
  fi
}

# Main execution
main() {
  local message="${1:-}"
  local activity_type="${2:-}"

  if [[ -z $message ]]; then
    log_error "No message provided"
    echo ""
    echo "Usage: $0 \"message\" [type]"
    echo ""
    echo "Types: general, feature, bug-fix, testing, documentation, refactoring, deployment, code-review, planning"
    echo ""
    echo "Example: $0 \"Fixed authentication bug in login module\" bug-fix"
    echo "Example: $0 \"Added new API endpoint for user profile\""
    exit 1
  fi

  # Auto-detect type if not provided
  if [[ -z $activity_type ]]; then
    activity_type=$(detect_activity_type "$message")
    log_info "Auto-detected activity type: $activity_type"
  fi

  # Log the activity
  log_activity "$message" "$activity_type"

  # Show current activity count
  if [[ -f $SESSION_FILE ]]; then
    local activity_count=$(jq '.activities | length' "$SESSION_FILE")
    echo -e "${BLUE}Session has $activity_count activities logged${NC}"
  fi
}

# Special command handling
case "${1:-}" in
  "--file")
    # Log file modification
    shift
    for file in "$@"; do
      log_file_modification "$file"
      log_info "Logged file modification: $file"
    done
    ;;
  "--command")
    # Log command execution
    shift
    log_command "$*"
    log_info "Logged command execution"
    ;;
  "--issue")
    # Log issue work
    shift
    issue_number="$1"
    issue_title="${2:-}"
    action="${3:-worked}"
    log_issue_work "$issue_number" "$issue_title" "$action"
    log_info "Logged work on issue #$issue_number"
    ;;
  "--help" | "-h")
    echo "Usage: $0 \"message\" [type]"
    echo "       $0 --file file1 [file2 ...]"
    echo "       $0 --command \"command string\""
    echo "       $0 --issue number [title] [action]"
    echo ""
    echo "Log activities to the current development session."
    echo ""
    echo "Activity Types:"
    echo "  general       - General development activity"
    echo "  feature       - New feature implementation"
    echo "  bug-fix       - Bug fixing"
    echo "  testing       - Test creation or execution"
    echo "  documentation - Documentation updates"
    echo "  refactoring   - Code refactoring"
    echo "  deployment    - Deployment activities"
    echo "  code-review   - Code review activities"
    echo "  planning      - Planning and design"
    echo ""
    echo "Special Commands:"
    echo "  --file        Log file modifications"
    echo "  --command     Log command execution"
    echo "  --issue       Log GitHub issue work"
    echo ""
    echo "The activity type is auto-detected if not specified."
    exit 0
    ;;
  *)
    # Regular activity logging
    main "$@"
    ;;
esac
