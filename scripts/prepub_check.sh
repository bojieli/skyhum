#!/usr/bin/env bash
# prepub_check.sh - fail if the tree holds audio, media, weights, data dumps, big files, .env files or key-like
# strings. Run before every push and before any publication.
#
#   scripts/prepub_check.sh            check tracked files (working tree)
#   scripts/prepub_check.sh --staged   check files staged for commit (pre-commit hook)
#   scripts/prepub_check.sh --history  also check every blob reachable from any ref (audio, weights, size)
#   scripts/prepub_check.sh --strict   also fail on private hostnames and absolute home paths (warnings otherwise)
#
# The patterns below are generic shapes of common API keys. No real key is, or should ever be, in this file.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2

MAX_BYTES=$((5 * 1024 * 1024))
MEDIA_RE='\.(wav|wave|mp3|flac|ogg|oga|opus|m4a|aac|wma|aif|aiff|caf|webm|mp4|m4v|mkv|mov|avi)$'
WEIGHT_RE='\.(pt|pth|ckpt|safetensors|onnx|pkl|pickle|joblib|h5|hdf5|npz|npy|parquet|sqlite|db|tar|tgz|zip|7z|gz|bz2|xz|zst)$'
ENV_RE='(^|/)\.env($|\.)|(^|/)[^/]*\.env$'
KEY_RE='AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}|xox[abprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|hf_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_-]{20}'
ASSIGN_RE='(api[_-]?key|apikey|secret|token|passwd|password|client[_-]?secret)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][A-Za-z0-9_/+=.-]{16,}["'"'"']'
PRIVATE_RE='intel[0-9]|epyc[0-9]|cloudml[0-9]?|rtx-pro|cherry ?servers?|mier-connect|vast\.ai|/home/[a-z]+/|/Users/[A-Za-z]+/'
SELF='scripts/prepub_check.sh'

mode=tree; strict=0; history=0
for a in "$@"; do
  case "$a" in
    --staged) mode=staged ;;
    --history) history=1 ;;
    --strict) strict=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

if [ "$mode" = staged ]; then
  files=$(git diff --cached --name-only --diff-filter=ACMR)
else
  files=$(git ls-files)
fi
# submodule gitlinks are not files
files=$(printf '%s\n' "$files" | while IFS= read -r f; do [ -f "$f" ] && printf '%s\n' "$f"; done)

fail=0; warn=0
bad() { echo "FAIL  $*"; fail=$((fail + 1)); }
note() { echo "WARN  $*"; warn=$((warn + 1)); }

while IFS= read -r f; do
  [ -z "$f" ] && continue
  lf=$(printf '%s' "$f" | tr '[:upper:]' '[:lower:]')
  if printf '%s' "$lf" | grep -Eq "$MEDIA_RE"; then bad "audio/video file: $f"; fi
  if printf '%s' "$lf" | grep -Eq "$WEIGHT_RE"; then bad "weights, data dump or archive: $f"; fi
  if printf '%s' "$f" | grep -Eq "$ENV_RE" && [ "$f" != ".env.example" ]; then bad ".env file: $f"; fi
  size=$(wc -c < "$f" | tr -d ' ')
  if [ "$size" -gt "$MAX_BYTES" ]; then bad "file over 5 MB ($size bytes): $f"; fi
  [ "$f" = "$SELF" ] && continue
  if grep -IEnq -e "$KEY_RE" "$f" 2>/dev/null; then
    grep -IEn -e "$KEY_RE" "$f" | cut -d: -f1 | head -3 | while read -r ln; do echo "FAIL  key-like string: $f:$ln"; done
    fail=$((fail + 1))
  fi
  if grep -IEinq -e "$ASSIGN_RE" "$f" 2>/dev/null; then
    grep -IEin -e "$ASSIGN_RE" "$f" | cut -d: -f1 | head -3 | while read -r ln; do echo "FAIL  secret-like assignment: $f:$ln"; done
    fail=$((fail + 1))
  fi
  if grep -IEiq -e "$PRIVATE_RE" "$f" 2>/dev/null; then
    n=$(grep -IEic -e "$PRIVATE_RE" "$f")
    if [ "$strict" = 1 ]; then bad "private host or home path ($n line(s)): $f"; else note "private host or home path ($n line(s)): $f"; fi
  fi
done <<< "$files"

if [ "$history" = 1 ]; then
  echo "scanning history of all refs..."
  git rev-list --all --objects | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  while read -r type _ size path; do
    [ "$type" = blob ] || continue
    lp=$(printf '%s' "$path" | tr '[:upper:]' '[:lower:]')
    if printf '%s' "$lp" | grep -Eq "$MEDIA_RE"; then echo "FAIL  history: audio/video blob: $path"; fi
    if printf '%s' "$lp" | grep -Eq "$WEIGHT_RE"; then echo "WARN  history: weights/dump blob: $path ($size bytes)"; fi
    if [ "$size" -gt "$MAX_BYTES" ]; then echo "FAIL  history: blob over 5 MB: $path ($size bytes)"; fi
  done | sort -u | tee /dev/stderr | grep -q '^FAIL' && fail=$((fail + 1))
fi

echo "prepub_check: $fail failure(s), $warn warning(s) ($mode$( [ $history = 1 ] && echo ', history')$( [ $strict = 1 ] && echo ', strict'))"
[ "$fail" -eq 0 ]
