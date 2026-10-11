#!/usr/bin/env bash
# One command to finish: verify, push, and fix the repository description.
#
#   bash finish.sh
#
# Safe to re-run. It never force-pushes and never deletes anything. If the push
# step fails it stops before touching the repository description, so you can
# retry from the same place.
set -euo pipefail

SLUG="Maloth04/Early-Prediction-of-Patient-Deterioration-in-the-ICU-An-Explainable-ML-Study-on-MIMIC-IV-and-eICU-"
REMOTE="https://github.com/${SLUG}.git"

DESCRIPTION="Explainable ML for early prediction of acute treatment escalation in the ICU — vasopressor initiation, invasive ventilation, or resuscitation, 4 and 8 hours ahead, from structured vitals and labs. Master's thesis code. Principal finding: a feature block containing no clinical values, only the record of when clinicians chose to measure, performs comparably to one containing all of them."

TOPICS='["clinical-prediction-model","intensive-care","mimic-iv","explainable-ai","shap","xgboost","informative-missingness","reproducible-research"]'

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mnote\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mstopped:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 0. are we in the right place ────────────────────────────────────────────
[ -f pyproject.toml ] && [ -d analysis ] && [ -d src/icu_deterioration ] \
  || die "run this from inside the icu-early-deterioration directory"
git rev-parse --git-dir >/dev/null 2>&1 || die "this directory is not a git repository"
ok "in $(pwd)"

# ── 1. nothing uncommitted ──────────────────────────────────────────────────
say "1/5  Checking the working tree"
if [ -n "$(git status --porcelain)" ]; then
  git status --short
  die "uncommitted changes above. Commit or stash them, then re-run."
fi
ok "clean, $(git rev-list --count HEAD) commit(s), $(git ls-files | wc -l | tr -d ' ') files"

# ── 2. the data-use check, locally, before anything leaves the machine ──────
say "2/5  Checking no patient data or credentials are tracked"
offenders=$(git ls-files \
  | grep -Ei '\.(csv|tsv|parquet|feather|dta|sav|pkl|joblib|pt|pth)$' \
  | grep -v '^results/' || true)
[ -z "$offenders" ] || { echo "$offenders"; die "data-shaped files tracked; the PhysioNet DUA forbids redistributing record-level data"; }
creds=$(git ls-files | grep -Ei '(^|/)(\.env|.*service-account.*\.json|.*credentials.*\.json)$' || true)
[ -z "$creds" ] || { echo "$creds"; die "credential-shaped files tracked"; }
ok "nothing to redistribute by accident"

# ── 3. tests ────────────────────────────────────────────────────────────────
say "3/5  Running the test suite"
PY=$(command -v python3 || command -v python) || die "no python on PATH"
if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
  warn "installing the test extras"
  "$PY" -m pip install -q -e ".[dev]"
fi
"$PY" -m pytest -q || die "tests failed; CI will fail too. Fix before pushing."
ok "tests pass, so CI will pass"

# ── 4. push ─────────────────────────────────────────────────────────────────
say "4/5  Pushing to GitHub"
if git remote get-url origin >/dev/null 2>&1; then
  ok "origin: $(git remote get-url origin)"
else
  git remote add origin "$REMOTE"
  ok "origin added"
fi
branch=$(git rev-parse --abbrev-ref HEAD)

# Read works anonymously on a public repo, so this separates "wrong URL" from
# "not signed in". GitHub answers an unauthorised push with 404, not 403, so it
# never reveals whether a repository exists — which is why a push 404 almost
# always means credentials, not a missing repository.
if ! git ls-remote origin >/dev/null 2>&1; then
  die "cannot reach $(git remote get-url origin)
  Check the URL. The repository name ends with a hyphen, and a copy-paste that
  drops it gives exactly this."
fi
ok "remote reachable, so the URL is right"

if [ -n "$(git ls-remote --heads origin 2>/dev/null)" ]; then
  warn "the remote already has branches — this will NOT force anything."
  warn "if the push is rejected, the remote has commits this history does not;"
  warn "decide deliberately rather than forcing."
fi

if ! git push -u origin "$branch"; then
  cat >&2 <<'HELP'

  The push failed. If the error said 404, that is authentication, not a missing
  repository: the repository is public and reachable, as the check above proved.
  GitHub answers an unauthorised push with 404 so it never reveals private repos.

  Easiest fix, on any operating system:

      gh auth login            # choose GitHub.com, HTTPS, authenticate in browser
      git push -u origin main

  Without the GitHub CLI, use a personal access token as the password:

      github.com -> Settings -> Developer settings -> Personal access tokens
        Classic token:      tick the whole "repo" scope
        Fine-grained token: under "Repository access" you MUST select this
                            repository, and give it Contents: Read and write

      git push -u origin main     # username: Maloth04, password: the token

  If it fails again without even asking for a password, a stale credential is
  cached. Clear it, then push again:

      macOS     git credential-osxkeychain erase <<< $'protocol=https\nhost=github.com\n'
      Windows   Credential Manager -> Windows Credentials -> remove git:https://github.com
      Linux     git config --global --unset credential.helper

HELP
  die "push failed, see above"
fi
ok "pushed $branch"

# ── 5. description and topics ───────────────────────────────────────────────
say "5/5  Fixing the repository description"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  # "cardiac arrest" is the term the thesis deliberately rejected: the structured
  # tables record a resuscitation response, not a confirmed arrest (Appendix D).
  gh api -X PATCH "repos/${SLUG}" -f description="$DESCRIPTION" >/dev/null \
    && ok "description replaced" \
    || warn "could not set the description; do it on the repository page"
  gh api -X PUT "repos/${SLUG}/topics" \
    -H "Accept: application/vnd.github+json" \
    --input - <<<"{\"names\": ${TOPICS}}" >/dev/null \
    && ok "topics set" \
    || warn "could not set topics; add them on the repository page"
else
  warn "no authenticated gh, so the description is unchanged."
  warn "it still says \"cardiac arrest\", which your thesis renames. Edit it at:"
  warn "  https://github.com/${SLUG}  (the gear beside \"About\")"
fi

say "Done"
cat <<EOF
  Repository   https://github.com/${SLUG}
  Actions      https://github.com/${SLUG}/actions
               expect: tests on 3.11 / 3.12 / 3.13, and "no patient data committed"

  Then, once:  Settings -> Branches -> protect ${branch}, require the CI checks.

  Read docs/reproducibility.md before the viva. The test suite found a band-boundary
  defect in news2_construction.py: Section 4.3.5's conclusions survive, two of its
  quoted figures do not. CHANGELOG.md has both states.
EOF
