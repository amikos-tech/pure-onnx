# Deferred Items

- Repository guidance: in a separate PR, add an `AGENTS.md` that points non-Claude agents to `CLAUDE.md` so the no-CGO rule and the rest of the canonical project guidance are not partially duplicated.
- OpenCLIP bootstrap: open a separate issue for `WithBootstrapCacheDir`, which validates `strings.TrimSpace(path)` but stores the original untrimmed path; cover the intended normalization behavior before changing it.
