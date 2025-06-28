#!/usr/bin/env bash
# Session Status Script - Show current session status with rich information
# Part of claude-init enhancement for persistent context tracking

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="${PWD}"
SESSION_DIR="${PROJECT_ROOT}/.claude"
SESSION_FILE="${SESSION_DIR}/session.json"
VERBOSE="${VERBOSE:-false}"

# Utility functions
log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Calculate session duration
calculate_duration() {
  local start_time="$1"
  local current_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  # Convert to seconds since epoch
  local start_seconds=$(date -d "$start_time" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$start_time" +%s 2>/dev/null || echo "0")
  local current_seconds=$(date +%s)

  if [[ $start_seconds -eq 0 ]]; then
    echo "unknown"
    return
  fi

  local duration=$((current_seconds - start_seconds))
  local hours=$((duration / 3600))
  local minutes=$(((duration % 3600) / 60))
  local seconds=$((duration % 60))

  if [[ $hours -gt 0 ]]; then
    echo "${hours}h ${minutes}m ${seconds}s"
  elif [[ $minutes -gt 0 ]]; then
    echo "${minutes}m ${seconds}s"
  else
    echo "${seconds}s"
  fi
}

# Get recent activities
get_recent_activities() {
  local session_data="$1"
  local limit="${2:-5}"

  echo "$session_data" | jq -r ".activities[-$limit:] | reverse | .[] | \"  • \\(.timestamp): \\(.description)\"" 2>/dev/null || echo "  • No activities recorded"
}

# Display session health metrics
display_health_metrics() {
  local session_data="$1"

  local health_start=$(echo "$session_data" | jq -r '.health_score_start // "0"')
  local health_current=$(echo "$session_data" | jq -r '.health_score_current // "0"')
  local health_change=$((health_current - health_start))

  echo -e "${PURPLE}📊 Health Metrics${NC}"
  echo -e "  Starting Score: $health_start/100"
  echo -e "  Current Score: $health_current/100"

  if [[ $health_change -gt 0 ]]; then
    echo -e "  Change: ${GREEN}+$health_change${NC} ✨"
  elif [[ $health_change -lt 0 ]]; then
    echo -e "  Change: ${RED}$health_change${NC} ⚠️"
  else
    echo -e "  Change: ${YELLOW}$health_change${NC} →"
  fi
  echo ""
}

# Display file modification summary
display_file_changes() {
  local session_data="$1"

  local files_count=$(echo "$session_data" | jq '.files_modified | length' 2>/dev/null || echo "0")

  if [[ $files_count -gt 0 ]]; then
    echo -e "${BLUE}📝 Files Modified${NC} ($files_count)"
    echo "$session_data" | jq -r '.files_modified[-5:] | .[] | "  • \(.)"' 2>/dev/null || true
    echo ""
  fi
}

# Display GitHub issues worked on
display_issues_worked() {
  local session_data="$1"

  local issues_count=$(echo "$session_data" | jq '.issues_worked | length' 2>/dev/null || echo "0")

  if [[ $issues_count -gt 0 ]]; then
    echo -e "${CYAN}🐛 Issues Worked${NC}"
    echo "$session_data" | jq -r '.issues_worked[] | "  • #\(.number): \(.title) [\(.status)]"' 2>/dev/null || true
    echo ""
  fi
}

# Get current git status update
get_current_git_status() {
  if [[ -d ".git" ]]; then
    local branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    local changes=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
    local staged=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')

    echo -e "${YELLOW}📋 Current Git Status${NC}"
    echo -e "  Branch: $branch"
    echo -e "  Total Changes: $changes files"
    echo -e "  Staged Changes: $staged files"

    # Show recent commits in this session
    local session_start=$(jq -r '.started' "$SESSION_FILE" 2>/dev/null || echo "")
    if [[ -n $session_start ]]; then
      local commits=$(git log --since="$session_start" --oneline 2>/dev/null | wc -l | tr -d ' ')
      if [[ $commits -gt 0 ]]; then
        echo -e "  Commits This Session: $commits"
        git log --since="$session_start" --oneline --max-count=3 2>/dev/null | sed 's/^/    /'
      fi
    fi
    echo ""
  fi
}

# Main status display
display_status() {
  if [[ ! -f $SESSION_FILE ]]; then
    log_error "No active session found"
    echo ""
    echo "Start a new session with: make session-start"
    exit 1
  fi

  local session_data=$(cat "$SESSION_FILE")
  local session_id=$(echo "$session_data" | jq -r '.id // "unknown"')
  local started=$(echo "$session_data" | jq -r '.started // "unknown"')
  local status=$(echo "$session_data" | jq -r '.status // "unknown"')
  local duration=$(calculate_duration "$started")

  # Header
  echo ""
  echo -e "${CYAN}📊 SESSION STATUS${NC}"
  echo "================================"
  echo ""

  # Basic Info
  echo -e "${GREEN}Session Information${NC}"
  echo -e "  ID: $session_id"
  echo -e "  Status: $status"
  echo -e "  Started: $started"
  echo -e "  Duration: $duration"
  echo ""

  # Project Info
  echo -e "${PURPLE}Project Context${NC}"
  echo -e "  Type: $(echo "$session_data" | jq -r '.project_info.type // "unknown"')"
  echo -e "  Language: $(echo "$session_data" | jq -r '.project_info.primary_language // "unknown"')"
  echo -e "  Frameworks: $(echo "$session_data" | jq -r '.project_info.frameworks // "none"')"
  echo ""

  # Health Metrics
  display_health_metrics "$session_data"

  # Current Git Status
  get_current_git_status

  # Recent Activities
  echo -e "${BLUE}📌 Recent Activities${NC}"
  get_recent_activities "$session_data" 5
  echo ""

  # Files Modified (if any)
  display_file_changes "$session_data"

  # Issues Worked (if any)
  display_issues_worked "$session_data"

  # Session Goals
  local goals_count=$(echo "$session_data" | jq '.goals | length' 2>/dev/null || echo "0")
  if [[ $goals_count -gt 0 ]]; then
    echo -e "${YELLOW}🎯 Session Goals${NC}"
    echo "$session_data" | jq -r '.goals[] | "  \(if .completed then "✅" else "⏳" end) \(.description)"' 2>/dev/null || true
    echo ""
  fi

  # Statistics Summary
  echo -e "${CYAN}📈 Session Statistics${NC}"
  echo -e "  Activities: $(echo "$session_data" | jq '.activities | length' 2>/dev/null || echo "0")"
  echo -e "  Files Modified: $(echo "$session_data" | jq '.files_modified | length' 2>/dev/null || echo "0")"
  echo -e "  Commands Executed: $(echo "$session_data" | jq '.commands_executed | length' 2>/dev/null || echo "0")"
  echo -e "  Issues Worked: $(echo "$session_data" | jq '.issues_worked | length' 2>/dev/null || echo "0")"
  echo ""

  # Next Actions
  echo -e "${GREEN}💡 Available Actions${NC}"
  echo '  • make session-log MSG="..."  - Log an activity'
  echo "  • make session-health         - Update health score"
  echo "  • make session-end           - End session with summary"
  echo ""
}

# JSON output mode
display_json() {
  if [[ ! -f $SESSION_FILE ]]; then
    echo '{"error": "No active session found"}'
    exit 1
  fi

  cat "$SESSION_FILE"
}

# Main execution
main() {
  local output_format="${1:-human}"

  case "$output_format" in
    "json")
      display_json
      ;;
    "human" | *)
      display_status
      ;;
  esac
}

# Help message
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  echo "Usage: $0 [format]"
  echo ""
  echo "Display current session status with rich information."
  echo ""
  echo "Arguments:"
  echo "  format    Output format: 'human' (default) or 'json'"
  echo ""
  echo "Environment variables:"
  echo "  VERBOSE=true    Enable verbose output"
  exit 0
fi

# Run main function
main "$@"
