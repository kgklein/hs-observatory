# Elsasser increment-PDF publication workflow

Run the three expensive analyses once:

```bash
hsobs-analyze analysis/malara-prod-000_elsasser-increment-pdf-publication.yaml
hsobs-analyze analysis/malara-prod-001_elsasser-increment-pdf-publication.yaml
hsobs-analyze analysis/malara-prod-002_elsasser-increment-pdf-publication.yaml
```

Then iterate on the figure using only those saved HDF5 products:

```bash
scripts/elsasser_increment_pdf_publication.sh
scripts/elsasser_increment_pdf_publication.sh --lags 100 1000 10000 \
  --output outputs/elsasser_increment_pdf_selected_lags.png
```

For a four-lag 2x2 figure:

```bash
scripts/elsasser_increment_pdf_publication.sh --layout-2x2 \
  --lags 100 500 2000 10000 \
  --output outputs/elsasser_increment_pdf_four_lags.png
```

The analysis reuses the existing local-pairwise perpendicular Elsasser
increment, geometric-mean normalization, and histogram implementation. The
publication YAML files use identical fixed lag and normalized-increment edges,
so PDFs are bin-aligned across truth and all MC members. The plotter verifies
this assumption before computing bin-wise MC 5th, 50th, and 95th percentiles.
Because the plotted quantity is the magnitude of a two-component perpendicular
increment divided by its geometric mean, the reference curve is the
geometric-mean-normalized Rayleigh distribution. The Wasserstein diagnostic
uses the same Rayleigh reference in its normalization.
Without `--lags`, it selects the second-smallest, middle, and second-largest
lag bins that are valid for every truth/MC dataset in all three products for
the 1x3 layout. The 2x2 layout automatically selects four well-spaced,
commonly valid lag bins.

To calculate PDFs only for three lags of interest, set the same option in all
three analysis YAML files before running the expensive stage:

```yaml
scale_bins:
  binning: log
  n_bins: 15
  lambda_min: 50.0
  lambda_max: 30000.0
  selected_lags_km: [100.0, 1000.0, 10000.0]
```

Each requested value is matched to the nearest center of the unchanged
15-bin lag grid. When exactly three bins were calculated, the plotter uses
them directly, so the normal plotting command is sufficient. They can also be
requested explicitly:

```bash
scripts/elsasser_increment_pdf_publication.sh --lags 100 1000 10000
```

This skips normalization, moment, sample, and histogram accumulation for all
other lag bins. Pair separations must still be evaluated to determine which
pairs enter the requested bins. With fixed `lambda_min` and `lambda_max`, the
analyzer also skips the former full pass used only to infer lag bounds.
