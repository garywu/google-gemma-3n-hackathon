#!/usr/bin/env bash
# Enhanced Session Start Script - Intelligent Session Management
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PWD}"
SESSION_DIR="${PROJECT_ROOT}/.claude"
SESSION_FILE="${SESSION_DIR}/session.json"
SESSION_HISTORY="${SESSION_DIR}/history"
CLAUDE_MD="${PROJECT_ROOT}/CLAUDE.md"
PROJECT_DETECTOR="${SCRIPT_DIR}/../project-detector.sh"

# Create necessary directories
mkdir -p "${SESSION_DIR}" "${SESSION_HISTORY}"

# Utility functions
log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Generate session ID
generate_session_id() {
  echo "session-$(date +%Y%m%d-%H%M%S)-$$"
}

# Get git status summary
get_git_status() {
  if [[ -d ".git" ]]; then
    local branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    local status=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
    local ahead_behind=$(git status -sb 2>/dev/null | grep -oE '\[.*\]' || echo "")
    local last_commit=$(git log -1 --pretty=format:'%h %s' 2>/dev/null || echo "No commits")

    echo "{
            \"branch\": \"$branch\",
            \"uncommitted_changes\": $status,
            \"sync_status\": \"$ahead_behind\",
            \"last_commit\": \"$last_commit\"
        }"
  else
    echo '{
            "branch": "none",
            "uncommitted_changes": 0,
            "sync_status": "not a git repo",
            "last_commit": "N/A"
        }'
  fi
}

# Run project analysis
run_project_analysis() {
  if [[ -f $PROJECT_DETECTOR ]]; then
    log_info "Running project analysis..."
    "$PROJECT_DETECTOR" "$PROJECT_ROOT" "json" 2>/dev/null || echo "{}"
  else
    echo "{}"
  fi
}

# Get workspace overview
get_workspace_overview() {
  local total_files=$(find . -type f -not -path "./.git/*" -not -path "./node_modules/*" 2>/dev/null | wc -l | tr -d ' ')
  local total_dirs=$(find . -type d -not -path "./.git/*" -not -path "./node_modules/*" 2>/dev/null | wc -l | tr -d ' ')
  local size=$(du -sh . 2>/dev/null | cut -f1 || echo "unknown")

  echo "{
        \"total_files\": $total_files,
        \"total_directories\": $total_dirs,
        \"size\": \"$size\"
    }"
}

# Check for active GitHub issues
get_github_issues() {
  if command -v gh &>/dev/null && [[ -d ".git" ]]; then
    gh issue list --limit 5 --json number,title,state,labels 2>/dev/null || echo "[]"
  else
    echo "[]"
  fi
}

# Initialize or load existing session
initialize_session() {
  local session_id=$(generate_session_id)
  local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local project_analysis=$(run_project_analysis)
  local git_status=$(get_git_status)
  local workspace_info=$(get_workspace_overview)
  local github_issues=$(get_github_issues)

  # Check for existing active session
  local previous_session_id=""
  if [[ -f $SESSION_FILE ]]; then
    previous_session_id=$(jq -r '.id // ""' "$SESSION_FILE" 2>/dev/null || echo "")

    # Archive previous session
    if [[ -n $previous_session_id ]]; then
      log_info "Archiving previous session: $previous_session_id"
      cp "$SESSION_FILE" "${SESSION_HISTORY}/${previous_session_id}.json"
    fi
  fi

  # Extract key project info
  local project_type=$(echo "$project_analysis" | jq -r '.project_type // "unknown"' 2>/dev/null || echo "unknown")
  local primary_language=$(echo "$project_analysis" | jq -r '.primary_language // "unknown"' 2>/dev/null || echo "unknown")
  local frameworks=$(echo "$project_analysis" | jq -r '.frameworks // ""' 2>/dev/null || echo "")
  local maturity_score=$(echo "$project_analysis" | jq -r '.maturity_score // "0"' 2>/dev/null || echo "0")

  # Create new session data
  local session_data=$(
    cat <<EOF
{
    "id": "$session_id",
    "started": "$timestamp",
    "status": "active",
    "previous_session": ${previous_session_id:+"\"$previous_session_id\""} ${previous_session_id:-null},
    "project_info": {
        "type": "$project_type",
        "primary_language": "$primary_language",
        "frameworks": "$frameworks",
        "maturity_score": $maturity_score,
        "path": "$PROJECT_ROOT"
    },
    "git_status": $git_status,
    "workspace": $workspace_info,
    "github_issues": $github_issues,
    "activities": [],
    "issues_worked": [],
    "files_modified": [],
    "commands_executed": [],
    "health_score_start": $maturity_score,
    "health_score_current": $maturity_score,
    "session_type": "development",
    "goals": [],
    "notes": []
}
EOF
  )

  # Write session file
  echo "$session_data" | jq '.' >"$SESSION_FILE"

  # Update CLAUDE.md with session info if it exists
  if [[ -f $CLAUDE_MD ]]; then
    update_claude_md "$session_id" "$timestamp" "$project_type" "$primary_language" "$frameworks" "$maturity_score"
  fi

  echo "$session_id"
}

# Update CLAUDE.md with session information
update_claude_md() {
  local session_id="$1"
  local timestamp="$2"
  local project_type="$3"
  local primary_language="$4"
  local frameworks="$5"
  local maturity_score="$6"

  # Create backup
  cp "$CLAUDE_MD" "${CLAUDE_MD}.backup"

  # Update using sed (simplified approach for reliability)
  # This is a basic update - in production, you'd use a more sophisticated approach
  log_info "Updating CLAUDE.md with session information..."

  # For now, just append session info if not already present
  if ! grep -q "Current Session ID: $session_id" "$CLAUDE_MD"; then
    echo -e "\n<!-- Session Info - Auto-updated by session management -->" >>"$CLAUDE_MD"
    echo -e "<!-- Current Session ID: $session_id -->" >>"$CLAUDE_MD"
    echo -e "<!-- Session Started: $timestamp -->" >>"$CLAUDE_MD"
  fi
}

# Display session summary
display_session_summary() {
  local session_id="$1"
  local session_data=$(cat "$SESSION_FILE")

  echo ""
  echo -e "${CYAN}🚀 ENHANCED SESSION STARTED${NC}"
  echo "================================"
  echo ""
  echo -e "${GREEN}Session ID:${NC} $session_id"
  echo -e "${GREEN}Started:${NC} $(echo "$session_data" | jq -r '.started')"
  echo ""

  echo -e "${PURPLE}📊 Project Intelligence${NC}"
  echo -e "  Type: $(echo "$session_data" | jq -r '.project_info.type')"
  echo -e "  Language: $(echo "$session_data" | jq -r '.project_info.primary_language')"
  echo -e "  Frameworks: $(echo "$session_data" | jq -r '.project_info.frameworks // "none"')"
  echo -e "  Maturity Score: $(echo "$session_data" | jq -r '.project_info.maturity_score')/100"
  echo ""

  echo -e "${BLUE}📁 Repository Status${NC}"
  echo -e "  Branch: $(echo "$session_data" | jq -r '.git_status.branch')"
  echo -e "  Uncommitted Changes: $(echo "$session_data" | jq -r '.git_status.uncommitted_changes')"
  echo -e "  Last Commit: $(echo "$session_data" | jq -r '.git_status.last_commit')"
  echo ""

  echo -e "${YELLOW}📋 Workspace Overview${NC}"
  echo -e "  Total Files: $(echo "$session_data" | jq -r '.workspace.total_files')"
  echo -e "  Total Directories: $(echo "$session_data" | jq -r '.workspace.total_directories')"
  echo -e "  Size: $(echo "$session_data" | jq -r '.workspace.size')"
  echo ""

  # Show active GitHub issues if any
  local issue_count=$(echo "$session_data" | jq '.github_issues | length')
  if [[ $issue_count -gt 0 ]]; then
    echo -e "${CYAN}🐛 Active GitHub Issues${NC}"
    echo "$session_data" | jq -r '.github_issues[] | "  #\(.number): \(.title)"' 2>/dev/null || true
    echo ""
  fi

  echo -e "${GREEN}💡 Session Commands${NC}"
  echo "  • make session-status    - Check current session"
  echo '  • make session-log MSG="..." - Log activity'
  echo "  • make session-end       - End session with summary"
  echo ""

  # Show recommendations based on project type
  case "$(echo "$session_data" | jq -r '.project_info.type')" in
    "web-app")
      echo -e "${YELLOW}🎯 Recommended Actions${NC}"
      echo "  • Run 'make dev-web' to start development server"
      echo "  • Use 'make lighthouse' for performance audit"
      echo "  • Check 'make build-web' before deployment"
      ;;
    "api")
      echo -e "${YELLOW}🎯 Recommended Actions${NC}"
      echo "  • Run 'make dev-api' to start API server"
      echo "  • Use 'make test-api' for integration tests"
      echo "  • Generate docs with 'make docs-api'"
      ;;
    "library")
      echo -e "${YELLOW}🎯 Recommended Actions${NC}"
      echo "  • Run 'make test' to verify functionality"
      echo "  • Use 'make build-lib' before publishing"
      echo "  • Update docs with 'make docs-lib'"
      ;;
  esac
  echo ""
}

# Main execution
main() {
  log_info "Initializing enhanced session tracking..."

  # Check if already in a session
  if [[ -f $SESSION_FILE ]]; then
    local existing_status=$(jq -r '.status // "unknown"' "$SESSION_FILE" 2>/dev/null || echo "unknown")
    if [[ $existing_status == "active" ]]; then
      log_warning "Active session detected. Starting new session will archive the current one."
      read -p "Continue? (y/N): " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Session start cancelled"
        exit 0
      fi
    fi
  fi

  # Initialize session
  local session_id=$(initialize_session)

  # Create session log entry
  local log_file="${SESSION_HISTORY}/${session_id}.log"
  echo "Session $session_id started at $(date)" >"$log_file"

  # Display summary
  display_session_summary "$session_id"

  log_success "Enhanced session tracking activated!"
}

# Help message
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  echo "Usage: $0"
  echo ""
  echo "Start an enhanced development session with intelligent tracking."
  echo ""
  echo "Features:"
  echo "  - Automatic project analysis and type detection"
  echo "  - Git repository status tracking"
  echo "  - GitHub issue integration"
  echo "  - Session persistence and history"
  echo "  - Context preservation across sessions"
  echo ""
  echo "Session data is stored in: .claude/session.json"
  echo "Session history is archived in: .claude/history/"
  exit 0
fi

# Run main function
main "$@"
