#!/usr/bin/env bash
# Session End Script - End session with comprehensive summary
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
SESSION_HISTORY="${SESSION_DIR}/history"
SUMMARY_DIR="${SESSION_DIR}/summaries"

# Create necessary directories
mkdir -p "${SUMMARY_DIR}"

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

# Calculate session duration
calculate_duration() {
  local start_time="$1"
  local end_time="$2"

  # Convert to seconds since epoch
  local start_seconds=$(date -d "$start_time" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$start_time" +%s 2>/dev/null || echo "0")
  local end_seconds=$(date -d "$end_time" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$end_time" +%s 2>/dev/null || echo "0")

  if [[ $start_seconds -eq 0 || $end_seconds -eq 0 ]]; then
    echo "unknown"
    return
  fi

  local duration=$((end_seconds - start_seconds))
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

# Update health score
update_health_score() {
  local session_data="$1"
  local project_detector="${SESSION_DIR}/../scripts/project-detector.sh"

  if [[ -f $project_detector ]]; then
    log_info "Updating health score..."
    local analysis=$("$project_detector" "$PROJECT_ROOT" "json" 2>/dev/null || echo "{}")
    local new_score=$(echo "$analysis" | jq -r '.maturity_score // "0"' 2>/dev/null || echo "0")
    echo "$new_score"
  else
    echo $(echo "$session_data" | jq -r '.health_score_current // "0"')
  fi
}

# Generate session summary
generate_summary() {
  local session_data="$1"
  local session_id=$(echo "$session_data" | jq -r '.id')
  local started=$(echo "$session_data" | jq -r '.started')
  local ended="$2"
  local duration="$3"
  local health_score_end="$4"

  local summary_file="${SUMMARY_DIR}/${session_id}-summary.md"

  cat >"$summary_file" <<EOF
# Session Summary: $session_id

**Duration**: $started → $ended ($duration)  
**Project**: $(echo "$session_data" | jq -r '.project_info.path')  
**Type**: $(echo "$session_data" | jq -r '.project_info.type')

## 📊 Session Metrics

| Metric | Value |
|--------|-------|
| **Activities Logged** | $(echo "$session_data" | jq '.activities | length') |
| **Files Modified** | $(echo "$session_data" | jq '.files_modified | length') |
| **Commands Executed** | $(echo "$session_data" | jq '.commands_executed | length') |
| **Issues Worked** | $(echo "$session_data" | jq '.issues_worked | length') |
| **Health Score Change** | $(echo "$session_data" | jq -r '.health_score_start')→$health_score_end |

## 🎯 Session Goals
EOF

  local goals_count=$(echo "$session_data" | jq '.goals | length')
  if [[ $goals_count -gt 0 ]]; then
    echo "$session_data" | jq -r '.goals[] | "- \(if .completed then "✅" else "❌" end) \(.description)"' >>"$summary_file"
  else
    echo "- No specific goals were set for this session" >>"$summary_file"
  fi

  cat >>"$summary_file" <<EOF

## 📝 Activities Summary

### By Type
EOF

  # Group activities by type
  echo "$session_data" | jq -r '
        .activities 
        | group_by(.type) 
        | map({
            type: .[0].type,
            count: length,
            activities: map(.description)
        })
        | .[]
        | "#### \(.type) (\(.count))\n" + (.activities | map("- " + .) | join("\n"))
    ' >>"$summary_file" 2>/dev/null || echo "No activities recorded" >>"$summary_file"

  # Files modified section
  local files_count=$(echo "$session_data" | jq '.files_modified | length')
  if [[ $files_count -gt 0 ]]; then
    cat >>"$summary_file" <<EOF

## 📁 Files Modified
EOF
    echo "$session_data" | jq -r '.files_modified[] | "- \(.)"' >>"$summary_file"
  fi

  # GitHub issues section
  local issues_count=$(echo "$session_data" | jq '.issues_worked | length')
  if [[ $issues_count -gt 0 ]]; then
    cat >>"$summary_file" <<EOF

## 🐛 GitHub Issues Worked
EOF
    echo "$session_data" | jq -r '.issues_worked[] | "- #\(.number): \(.title) [\(.status)]"' >>"$summary_file"
  fi

  # Git commits in session
  if [[ -d ".git" ]]; then
    local commits=$(git log --since="$started" --until="$ended" --oneline 2>/dev/null)
    if [[ -n $commits ]]; then
      cat >>"$summary_file" <<EOF

## 💾 Commits Made
EOF
      echo "$commits" | sed 's/^/- /' >>"$summary_file"
    fi
  fi

  # Key takeaways
  cat >>"$summary_file" <<EOF

## 🔑 Key Takeaways

### What went well
$(echo "$session_data" | jq -r '.notes[] | select(.type == "success") | "- \(.content)"' 2>/dev/null || echo "- Session completed successfully")

### Challenges faced
$(echo "$session_data" | jq -r '.notes[] | select(.type == "challenge") | "- \(.content)"' 2>/dev/null || echo "- No major challenges recorded")

### Next session priorities
$(echo "$session_data" | jq -r '.notes[] | select(.type == "next") | "- \(.content)"' 2>/dev/null || echo "- Continue development as planned")

---

*Generated by claude-init Enhanced Session Tracking*
EOF

  echo "$summary_file"
}

# Archive session
archive_session() {
  local session_data="$1"
  local session_id=$(echo "$session_data" | jq -r '.id')
  local ended="$2"

  # Update session with end time and status
  local temp_file="${SESSION_FILE}.tmp"
  jq --arg ended "$ended" \
    '.ended = $ended | .status = "completed"' \
    "$SESSION_FILE" >"$temp_file"
  mv "$temp_file" "$SESSION_FILE"

  # Archive to history
  local archive_file="${SESSION_HISTORY}/${session_id}.json"
  cp "$SESSION_FILE" "$archive_file"

  # Remove active session file
  rm "$SESSION_FILE"

  log_success "Session archived to: $archive_file"
}

# Display end summary
display_end_summary() {
  local summary_file="$1"
  local session_id="$2"
  local duration="$3"
  local activities_count="$4"
  local files_count="$5"
  local health_change="$6"

  echo ""
  echo -e "${CYAN}🏁 SESSION ENDED${NC}"
  echo "================================"
  echo ""
  echo -e "${GREEN}Session ID:${NC} $session_id"
  echo -e "${GREEN}Duration:${NC} $duration"
  echo ""

  echo -e "${PURPLE}📊 Session Statistics${NC}"
  echo "  • Activities: $activities_count"
  echo "  • Files Modified: $files_count"

  if [[ $health_change -gt 0 ]]; then
    echo -e "  • Health Score: ${GREEN}+$health_change${NC} ✨"
  elif [[ $health_change -lt 0 ]]; then
    echo -e "  • Health Score: ${RED}$health_change${NC} ⚠️"
  else
    echo -e "  • Health Score: ${YELLOW}No change${NC}"
  fi
  echo ""

  echo -e "${BLUE}📋 Summary saved to:${NC}"
  echo "  $summary_file"
  echo ""

  echo -e "${YELLOW}💡 Next Steps${NC}"
  echo "  • Review the session summary"
  echo "  • Commit any uncommitted changes"
  echo "  • Update project documentation if needed"
  echo "  • Start a new session when ready"
  echo ""
}

# Main execution
main() {
  if [[ ! -f $SESSION_FILE ]]; then
    log_error "No active session found"
    exit 1
  fi

  log_info "Ending current session..."

  # Read session data
  local session_data=$(cat "$SESSION_FILE")
  local session_id=$(echo "$session_data" | jq -r '.id')
  local started=$(echo "$session_data" | jq -r '.started')
  local ended=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  # Calculate metrics
  local duration=$(calculate_duration "$started" "$ended")
  local health_score_start=$(echo "$session_data" | jq -r '.health_score_start // "0"')
  local health_score_end=$(update_health_score "$session_data")
  local health_change=$((health_score_end - health_score_start))

  # Get counts
  local activities_count=$(echo "$session_data" | jq '.activities | length')
  local files_count=$(echo "$session_data" | jq '.files_modified | length')

  # Generate summary
  local summary_file=$(generate_summary "$session_data" "$ended" "$duration" "$health_score_end")

  # Archive session
  archive_session "$session_data" "$ended"

  # Display summary
  display_end_summary "$summary_file" "$session_id" "$duration" "$activities_count" "$files_count" "$health_change"

  log_success "Session ended successfully!"
}

# Handle options
case "${1:-}" in
  "--help" | "-h")
    echo "Usage: $0 [--force]"
    echo ""
    echo "End the current development session with a comprehensive summary."
    echo ""
    echo "Options:"
    echo "  --force    End session without confirmation"
    echo ""
    echo "The session summary includes:"
    echo "  - Duration and timeline"
    echo "  - Activities performed"
    echo "  - Files modified"
    echo "  - GitHub issues worked"
    echo "  - Commits made"
    echo "  - Health score changes"
    echo "  - Key takeaways"
    exit 0
    ;;
  "--force")
    main
    ;;
  *)
    # Confirm before ending
    if [[ -f $SESSION_FILE ]]; then
      local session_id=$(jq -r '.id' "$SESSION_FILE")
      echo -e "${YELLOW}About to end session: $session_id${NC}"
      read -p "Are you sure? (y/N): " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        main
      else
        log_info "Session end cancelled"
      fi
    else
      log_error "No active session found"
    fi
    ;;
esac
