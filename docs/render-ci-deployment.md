# Render CI-controlled deployment

Production deploys are controlled by GitHub Actions, not Render auto-deploy.

## Required GitHub secrets

- `RENDER_DEPLOY_HOOK_URL`: Render deploy hook for the existing production service.
- `RENDER_PUBLIC_BASE_URL`: public production base URL, for example `https://fraudguard-api.onrender.com`.
- `FRAUDGUARD_API_KEY`: production API key used by the post-deploy smoke test.

## Required GitHub environment

Use the `production` environment for the Render deployment workflow. Keep any
approval or reviewer protection configured on that environment.

## Render dashboard steps

Do not delete or recreate the live Render service. On the existing service:

1. Disable Render auto deploy for the connected Git repository.
2. Confirm the service uses the root `Dockerfile`.
3. Confirm the health check path is `/live`.
4. Keep secrets in Render environment variables or secret fields only.
5. Confirm the service exposes Render's `RENDER_GIT_COMMIT` runtime metadata so
   `/version` can verify the exact deployed commit.

`render.yaml` also sets `autoDeploy: false` for future blueprint syncs.
