# External Integrations

**Analysis Date:** 2026-03-18

## APIs & External Services

**ONNX Runtime Releases:**
- Microsoft GitHub releases - primary source for native runtime bootstrap in `ort/bootstrap.go`
  - SDK/Client: standard `net/http`
  - Auth: optional `GITHUB_TOKEN` / `GH_TOKEN` for GitHub API rate-limit headroom
  - Endpoints used: release metadata under `api.github.com/repos/microsoft/onnxruntime/releases/tags/*` and release asset downloads under `github.com/microsoft/onnxruntime/releases/download/*`

**Model Hosting:**
- Hugging Face Hub - source for OpenCLIP assets and several test fixtures
  - Integration method: direct HTTPS downloads via `net/http` in `embeddings/openclip/bootstrap.go` and test helpers in `embeddings/*`
  - Auth: optional `HF_TOKEN` bearer token for gated/private assets
  - Artifacts used: `text_model.onnx`, `vision_model.onnx`, `tokenizer.json`, `preprocessor_config.json`, hosted parity datasets

**Release Distribution:**
- Cloudflare R2 - release artifact storage configured in `.github/workflows/release.yml`
  - SDK/Client: AWS CLI over S3-compatible API
  - Auth: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`
  - Objects used: release archives, `SHA256SUMS`, `latest.json`, optional `releases.json`

## Data Storage

**Databases:**
- None - the library is stateless and does not persist application data to a database.

**File Storage:**
- Local filesystem caches under the user cache dir - bootstrap artifacts for ONNX Runtime and OpenCLIP.
  - ORT cache: default user cache under `onnx-purego/onnxruntime` from `ort/bootstrap.go`
  - OpenCLIP cache: default user cache under `onnx-purego/openclip` from `embeddings/openclip/bootstrap.go`

**Caching:**
- In-process LRU caches for batch-size-specific sessions in `embeddings/minilm/embedder.go`, `embeddings/splade/embedder.go`, and `embeddings/openclip/embedder.go`
- GitHub Actions cache for downloaded model assets in `.github/workflows/ci.yml`

## Authentication & Identity

**Auth Provider:**
- None for library consumers - there is no user/session auth inside the Go packages.

**Token-Based Integrations:**
- GitHub API tokens for bootstrap metadata lookup and CI
- Hugging Face token for model/dataset downloads and optional private artifact access
- Cloudflare and R2 credentials only inside release automation

## Monitoring & Observability

**Error Tracking:**
- None integrated into the library itself; errors are returned directly to callers.

**Analytics:**
- None

**Logs:**
- Standard output/error only
  - Library code logs a small number of warnings via `log.Printf` in bootstrap and environment initialization paths
  - CI logs come from GitHub Actions workflow steps

## CI/CD & Deployment

**Hosting:**
- GitHub repository plus GitHub Actions for CI
- Release artifacts mirrored to Cloudflare R2 and GitHub Releases

**CI Pipeline:**
- GitHub Actions workflows in `.github/workflows/`
  - `ci.yml` - lint, cross-platform tests, targeted race tests, real-model integration tests
  - `release.yml` - build, sign, verify, and publish release bundles
  - `claude.yml` and `claude-code-review.yml` - repo automation around Claude Code
  - `dependabot.yml` - dependency update automation

## Environment Configuration

**Development:**
- No `.env` contract; configuration is done through exported environment variables and CLI tooling.
- Developers commonly need `ONNXRUNTIME_LIB_PATH` for explicit local runtime setup.
- Network access is optional for normal unit tests, but required for bootstrap/download-based flows.

**Staging:**
- No distinct staging environment exists in the codebase.

**Production:**
- Consumer applications are expected to provide the correct ONNX Runtime shared library or rely on the bootstrapper.
- Release publishing requires GitHub Secrets and optional Cloudflare cache-purge variables defined in the repository settings.

## Webhooks & Callbacks

**Incoming:**
- None in the application/library runtime.

**Outgoing:**
- None in the library runtime beyond direct HTTPS downloads for bootstrapping and test asset retrieval.

---

*Integration audit: 2026-03-18*
*Update when adding or removing external services*
