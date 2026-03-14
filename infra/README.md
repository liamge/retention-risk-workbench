# Deployment & Ops Playbook (Demo‑Only)

This folder adds lightweight artifacts to demonstrate enterprise-grade deployment of the churn inference API. They are intentionally simplified for portfolio/demos and are **not** hardened for internet-facing production.

## 1) Container image
- `infra/Dockerfile` builds a slim FastAPI image with model artifacts mounted at runtime.
- Build and test locally:
  ```bash
  cd infra
  docker build -t churn-risk-api:latest -f Dockerfile ..
  docker run --rm -p 8000:8000 -e API_KEY=dev-key -v ../artifacts:/app/artifacts:ro churn-risk-api:latest
  ```

## 2) Local stack with observability
- `infra/docker-compose.yml` brings up the API, MLflow tracking, Prometheus, and Grafana.
- Start everything:
  ```bash
  cd infra
  docker compose up --build
  ```
- Prometheus scrapes `/metrics` (enabled via `prometheus-fastapi-instrumentator`).
- Grafana (localhost:3000, admin/admin) can import a Prometheus datasource pointing to `http://prometheus:9090` and dashboards for latency/error-rate alerts.
- Demo caveats: credentials are defaults; no TLS, network policies, or persistence hardening included.
- MLflow now mounts your repo's `mlruns/` directory into the container and uses it as a file store backend (`file:///mlruns`), so your existing runs/experiments appear in the UI (`http://localhost:5000`).
- New `dashboard` service builds a separate image (Dockerfile.dashboard) and exposes Streamlit at `http://localhost:8501`.

## 3) Kubernetes deployment
- `infra/k8s/deployment.yaml` includes Deployment + Service + HPA + Ingress.
- Secrets: create `churn-api-secrets` with `api-key` before applying.
- Apply manifests after pushing an image to your registry (replace `ghcr.io/your-org/churn-risk-api:latest`).
  ```bash
  kubectl apply -f infra/k8s/deployment.yaml
  ```
- Ingress is TLS-enabled; terminate with cert-manager/ACM/Cloud DNS as appropriate.
- Demo caveats: manifests omit pod security standards, network policies, mTLS, image signing, and secret management; add before real traffic.

## 4) Managed cloud endpoints (pick one)
- **AWS SageMaker**: push the Docker image to ECR, then create a SageMaker EndpointConfig + Endpoint. Expose `/predict` for real-time scoring; attach IAM role with S3/CloudWatch access; enable auto-scaling with Application Auto Scaling based on InvocationsPerInstance.
- **GCP Vertex AI**: upload the image to Artifact Registry and deploy a `CustomPredictionRoutine` to an endpoint; configure VPC-SC if needed and use Cloud Monitoring/Alerting on `prediction/latency` and error metrics.
- **Azure ML**: register the image as an `OnlineEndpoint` with a blue/green traffic split for canarying.
- Demo caveats: high-level guidance only; production requires org policies, approvals, and guardrails.

## 5) Authentication & access control
- API-key guard is baked into `api/main.py` (header `X-API-Key`). Set `API_KEY` in env/secret to require it.
- In production, front the service with an API gateway (Kong/Apigee/ALB + Cognito/Auth0) for OIDC/JWT verification, rate limits, and WAF rules. Keep service-level key as defense-in-depth.
- Demo caveats: no rate limiting, DDoS protections, or secret rotation provided here.

## 6) Monitoring & alerting
- `/metrics` exposes request counts, latency histograms, and error codes.
- Prometheus rules live in `infra/monitoring/alert.rules.yml` (p95 latency and error-rate). Wire alerts to PagerDuty/Slack via Alertmanager.
- Application logs should be in JSON (uvicorn default with `--log-config`). Ship to CloudWatch Logs / GCP Logging / Elastic with fluent-bit or sidecar.
- Add drift monitoring by sending feature distributions to MLflow or Evidently on a schedule; alert when PSI/KL divergence crosses thresholds.
- Demo caveats: alert routing, SLOs, and runbooks are not included.

## 7) CI/CD outline
A typical GitHub Actions pipeline:
- lint & unit tests
- train (or load) model artifact stub for contract tests
- build & scan image (Trivy/grype)
- push to registry
- deploy to staging k8s or managed endpoint, run smoke test against `/health` and `/predict`
- manual or automated promotion to prod
- Demo caveats: supply-chain security (SBOMs, attestations), approvals, and secrets management are out of scope.

## 8) Configuration & secrets
- Keep runtime config in environment variables; no secrets in git.
- Use `AWS Secrets Manager` / `GCP Secret Manager` or Kubernetes Secrets sealed with `sealed-secrets`.
- Mount `artifacts/` read-only; version models via MLflow registry or object-store paths.
- Demo caveats: secrets are plaintext in local examples—replace with managed secret stores.

## 9) Backup / recovery
- Model artifacts stored in object storage (S3/GCS/Azure Blob) with lifecycle policies.
- MLflow backend DB should have backups (RDS/CloudSQL) and PITR enabled.

Use these artifacts as conversation starters in interviews or PRs: they show end-to-end thinking without locking you into a specific cloud. Do **not** expose them to production traffic without proper hardening and governance.
