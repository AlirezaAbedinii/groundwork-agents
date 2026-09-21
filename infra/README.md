# infra

Infrastructure as code lands here in Phase 4: a Terraform root module under
`infra/terraform/` for the Azure deployment (resource group, Postgres Flexible
Server with the `vector` extension, Container Apps for the RAG and agent APIs,
Key Vault, Application Insights, remote state in Azure Storage), driven by
`make up` / `make down`, with images on ghcr.io deployed from GitHub Actions via
OIDC. Empty until then.
