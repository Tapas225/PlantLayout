# Decisions and unresolved assumptions

## Initial decisions

- Example list C is the first development dataset.
- Source inputs are kept unchanged under `data/raw/`.
- The grid cell size is 0.2 m by 0.2 m.
- The target is 500 plants, subject to the capacity search.
- Experiment seeds are fixed and recorded in configuration.
- Input paths are repository-relative.
- The long-format trait table remains the source of truth; repeated values
  are preserved rather than deduplicated.
- Missing planting distances remain missing until the documented fallback is
  selected.
- The initial geometry benchmark uses a 5 m by 10 m outer bounding box
  (1,250 cells), with a 0.4 m vertical path or a centred 2 m by 2 m hole.
- Path and hole geometries retain the rectangle's outer bounding box; only
  plantable-cell count changes.
- Boundary clearance uses a one-cell non-plantable padding before the
  Euclidean distance transform.

## Step 1 data validation

- The CSV contains 17 taxa and 1,006 trait rows.
- The detailed report plant list contains 17 taxa and 100 plants.
- Missing lower planting distance values occur for `Cichorium intybus var.
  sativum` and `Epilobium angustifolium 'Album'`.
- Multiple sociability intervals occur for `Ajuga reptans 'Stölzle'`,
  `Centaurea jacea 'Caramia' - R -`, and `Hieracium bornmuelleri`; all
  intervals are retained.
- Relative quantities for a requested total use largest-remainder allocation;
  ties are resolved deterministically by taxon name.
- The disc-area screen interprets CSV planting distances as centimetres and
  converts them to metres.
- Capacity search uses bisection under an explicit monotonic-feasibility
  assumption and verifies the counts immediately below and above the result.
- Rule penalties are returned separately and remain in the interval [0, 1];
  weights are applied through `EnergyBreakdown.weighted()`.
- Sociability entries are interpreted as the union of all intervals recorded
  for a taxon, not as a single deduplicated value.
- Greedy tie-breaking uses the run seed; candidate energy evaluations are
  counted individually for later budget matching.
- Annealing currently uses exact full-energy recomputation for correctness;
  `energy_delta()` is the verification reference before introducing a
  spatially incremental evaluator.
- Experiment records use one row per geometry, seed, and method so paired
  comparisons remain reproducible and can be exported without recomputation.
- Confirmed fallback: `Cichorium intybus var. sativum` uses 50 cm from
  `Cichorium intybus var. intybus`.
- Unresolved: `Epilobium angustifolium 'Album'` requires supervisor
  confirmation and is not assigned a fallback.
- Interim pilot choice: exclude `Epilobium angustifolium 'Album'` explicitly,
  reducing the pilot list from 100 to 97 plants; this is not the final
  research dataset.
- A bounded 8-plant smoke test completed on all three geometries. The
  97-plant pilot was stopped after exceeding four minutes because full
  energy recomputation makes greedy candidate evaluation expensive; the
  spatially incremental evaluator is therefore required before large runs.
- The evaluator abstraction was added without changing scores. Local cached
  rule terms remain the next performance task.
- Profiling showed trait parsing was a measurable repeated cost; identity
  caches were added for sociability, height, and bloom-month features. The
  larger local-neighbourhood cache is still pending.
- Hard-constraint checks were vectorized after profiling; the bounded
  8-plant profile improved from about 3.0 s to 2.5 s with unchanged tests.
- Height neighborhoods now use one distance matrix per candidate instead of
  recomputing distances for every plant; the bounded profile remains about
  2.4 s with unchanged scores.

## To confirm before experiments

- Whether the three geometries share one outer bounding box or one plantable
  area.
- Whether the lower planting-distance value is the required minimum in mixed
  plantings.
- Which related taxon supplies each missing spacing fallback.
- Exact interpretation of sociability intervals when a taxon has duplicate
  or disjoint intervals.
- Whether a viewing direction is required for the height rule.
- The weights for the four soft rules.

Every change to these assumptions should be recorded here with its rationale
and the affected experiment.
