# Spike Conventions

## Artifact Shape

- Keep experiments isolated under `.planning/spikes/`; do not change production
  packages to prove an idea.
- Use a zero-padded number and kebab-case name for a standalone spike.
- Use one shared number with letter suffixes for head-to-head variants.
- Give every variant a `README.md` with frontmatter, the exact hypothesis,
  primary research, run commands, investigation trail, measured results, and
  verdict.
- Record comparison decisions in a sibling `NNN-*-comparison.md` artifact.

## Verification

- Prefer an executable proof over prose alone.
- Run concurrency-sensitive Go prototypes under `go test -race`.
- Record actual command output and benchmark allocations; do not turn small
  benchmark differences into a decision unless the code path is material.
- For purego FFI, split verification into:
  1. instrumented callbacks under `-race` for ownership and exact accounting;
  2. a real native ABI round trip without `-race`.

The split is required because the project's intentional `uintptr` FFI boundary
is incompatible with the checkptr mode enabled by race builds. Do not disable
checkptr merely to make one combined test pass.

## Decision Discipline

- Compare public API and maintenance burden before implementation detail.
- Reuse standard-library contracts and no-op implementations when they satisfy
  the requirement.
- Keep spike-only helpers private unless the public shape itself is what the
  experiment validates.
- State what a type cannot enforce. For example, a logging interface cannot
  prevent a call site from logging an error that is also returned; that rule
  needs a call-site audit and tests.
