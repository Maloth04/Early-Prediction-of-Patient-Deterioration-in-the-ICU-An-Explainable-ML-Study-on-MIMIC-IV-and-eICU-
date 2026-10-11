# Security and data protection

## Reporting

For a vulnerability in this code, open a [private security advisory](../../security/advisories/new)
rather than a public issue.

## The real risk here

This repository contains no patient data and no credentials, and the greater risk is
that someone adds some. If you find **anything resembling record-level MIMIC-IV or
eICU data, or a credential, in this repository or its history**, please report it
privately and immediately rather than opening a public issue.

Protections in place:

- `.gitignore` excludes every tabular and serialised format by default.
- CI fails the build if a data-shaped or credential-shaped file is tracked.
- `docs/data-access.md` states what may and may not be stored.

These are safeguards, not permission. The PhysioNet Credentialed Health Data Use
Agreement governs the data; obtain your own access rather than seeking a copy.

## Not for clinical use

Nothing here is a medical device or validated for clinical use. The thesis reports
that geographic external validation was not delivered, that the endpoint records a
treatment decision rather than a patient state, and that the model was never
prospectively evaluated.
