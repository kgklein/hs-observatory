# Documentation Development

The site uses MkDocs and Material for MkDocs. Existing scientific reference pages
stay in `docs/`; the root README is a concise landing page; DESIGN remains the repository
design authority. Workflow guides link canonical scientific references. No API-autogeneration plugin is required.

## Preview the site locally

From the repository root, create a Python environment if you do not already have
one (`python3 -m venv .venv`), then run:

```bash
source .venv/bin/activate
python -m pip install -e '.[docs]'
python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/hs-observatory/> in your browser, or follow the URL
printed in the terminal. The `/hs-observatory/` prefix comes from `site_url` in
`mkdocs.yml`. Keep the terminal command running while browsing. Saved edits
reload automatically; press **Ctrl+C** to stop the server. On later visits, activate
the environment and run the serve command again; reinstalling is unnecessary.

If port 8000 is busy, use `--dev-addr 127.0.0.1:8001` and open the printed URL.
If the command runs on a remote machine, forward its port 8000 to your computer
before opening the URL in your local browser.

## Build and diagram checks

The checked-in diagrams are sufficient for a local preview. To regenerate and
check the architecture diagrams, run:

```bash
python3 scripts/generate_architecture_graph.py
python3 -m unittest discover -s tests -p test_architecture_graph.py
```

For a release-style check:

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

## Content ownership and migration checks

Scientific equations/parameters/products belong to their method reference;
shared schema belongs to the data reference.
Root DESIGN/CONTRIBUTING and visualization helpers remain separate authorities.

Before retiring a legacy page
check local links/anchors/assets, and retain practical compatibility links.
Strict builds validate site links but cannot establish scientific correctness or
public availability of external inputs.

### Research-note navigation and source hygiene

Keep the workflow sections in `mkdocs.yml` and use the
[Research Notes landing page](../research-notes/index.md) to link detailed reports.
Only a small selection belongs in the sidebar. Add intentionally unlisted pages
to `not_in_nav`; they are still built, searched, and checked for broken links.
MkDocs 1.6 supports this setting. Navigation omissions are warnings so newly
added pages require an explicit navigation decision.

Use `exclude_docs` only for material that should not be published. The legacy
`architecture/README.md` conflicts with that directory's maintained `index.md`
and remains excluded. Editor backups (`*~`) are excluded from the site as well
as ignored by Git; MkDocs does not apply `.gitignore` to documentation sources.
Keep figures, equation SVGs, and architecture diagrams with their source pages
and check relative paths after moving a report.

## Equation rendering without new dependencies

Display and inline TeX equations in timing/gradient references have local SVG
renderings made with the existing Matplotlib dependency. Their original TeX is
preserved in expandable source blocks (display equations) and
`docs/assets/equations/sources.json` (all equations). No JavaScript renderer,
external CDN or additional documentation dependency is required. Mathematical
fenced examples elsewhere remain literal code/text as originally written.

After an approved equation edit, update its source record and regenerate from
the repository root using the existing environment:

```python
import json
from pathlib import Path
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image

root = Path("docs/assets/equations")
for equation in json.loads((root / "sources.json").read_text()):
    math_to_image(
        "$" + equation["mathtext"] + "$",
        root / equation["asset"],
        prop=FontProperties(size=14 if equation["display"] else 11),
        format="svg",
    )
```

The rendering spelling `\frac{1}{2}` replaces `\tfrac12` only in mathtext input;
the original TeX remains in the source record. Review renderings for clipping and
sign/index fidelity; a successful build alone cannot validate algebra.
