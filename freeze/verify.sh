#!/usr/bin/env bash
# Check clips/, results/, data/ and semantics/ against the results-freeze-1
# checksums. Run from the repository root. Prints only failures; exits non-zero
# if any file is missing or changed.
set -u
cd "$(dirname "$0")/.."
if command -v sha256sum >/dev/null 2>&1; then check="sha256sum -c --quiet"; else check="shasum -a 256 -c --quiet"; fi
status=0
for f in freeze/checksums/*.sha256; do
  if $check "$f"; then echo "ok      $f ($(wc -l < "$f" | tr -d ' ') files)"; else echo "FAILED  $f"; status=1; fi
done
exit $status
