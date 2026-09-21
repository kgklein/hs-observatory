# Documentation Development

The site uses MkDocs and Material for MkDocs. Existing scientific reference pages
stay in `docs/`; the root README and DESIGN remain repository documents, with
concise site-facing summaries. No API-autogeneration plugin is required.

## Local commands

From the repository root in an activated Python environment:

```bash
python3 -m pip install -e ".[docs]"
python3 scripts/generate_architecture_graph.py
python3 -m unittest discover -s tests -p test_architecture_graph.py
mkdocs serve
```

Follow the URL printed by `mkdocs serve`. For a release-style check:

```bash
mkdocs build --strict
```

`python3 -m mkdocs serve` and `python3 -m mkdocs build --strict` are equivalent
when the commands are not on PATH. Generated HTML goes into the ignored `site/`
directory. Graph generation requires only Python's standard library; install
Graphviz using your platform's package manager to regenerate SVGs. Without `dot`,
DOT generation succeeds and existing SVGs are left unchanged.

Use relative links between documentation pages so assets and navigation work at
the intended `https://kgklein.github.io/hs-observatory/` subpath. Root repository
documents and configurations can be linked through their GitHub URLs; those
links require repository access before public release.

## Build validation now

`.github/workflows/docs.yml` runs on relevant pull requests, pushes to `main`,
and manual dispatch. It installs the documentation extra and Graphviz, runs
focused generator tests, regenerates both diagrams, and builds with `--strict`.
Source changes also trigger validation so diagrams can follow package changes.
The workflow has read-only repository permissions and does not publish, upload
a Pages artifact, or require Pages to be enabled.

## Enable GitHub Pages later

Do this only when the repository and documentation are ready for public release.
No deployment workflow is active now. The intended URL is already set in
`mkdocs.yml`; changing repository visibility or settings is a separate maintainer
decision.

1. In GitHub, open **Settings → Pages → Build and deployment → GitHub Actions**.
2. Update `.github/workflows/docs.yml` with the additions below. Keep the current
   validation triggers and build steps.
3. Commit/review those changes through the normal project process, then merge to
   `main` or manually run the workflow on `main`.
4. Confirm the deployment URL and review the published navigation and figures.

Add job-level permissions to `jobs.build` so the configuration step can read the
Pages settings (the current validation-only job does not need this):

```yaml
permissions:
  contents: read
  pages: read
```

Add this step in `jobs.build.steps` before the strict build:

```yaml
- name: Configure Pages
  if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
  uses: actions/configure-pages@v6
```

Add this step after the strict build:

```yaml
- name: Upload Pages artifact
  if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
  uses: actions/upload-pages-artifact@v5
  with:
    path: site
```

Add this job alongside `build`, under the existing `jobs` mapping:

```yaml
deploy:
  if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'
  needs: build
  runs-on: ubuntu-latest
  permissions:
    pages: write
    id-token: write
  concurrency:
    group: github-pages
    cancel-in-progress: false
  environment:
    name: github-pages
    url: ${{ steps.deployment.outputs.page_url }}
  steps:
    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v5
```

The Pages write and OIDC permissions are scoped to the new deploy job. Pull requests
continue to build without deploying. This uses the official Pages artifact
deployment mechanism, not a `gh-pages` branch. See GitHub's
[custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [deployment action](https://github.com/actions/deploy-pages) for the settings
and permission requirements. Check official action release notes when enabling
publication in the future.
