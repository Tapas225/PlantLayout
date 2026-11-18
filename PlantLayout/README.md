# PlantLayout

Reproducible implementation of the research project *Rule-Based Optimization
of Perennial Planting Plans by Greedy Insertion and Simulated Annealing*.

## Project layout

- `config/`: configuration files and experiment settings
- `data/raw/`: source inputs copied unchanged from the supplied files
- `notes/`: research notes and rule definitions
- `results/`: generated outputs; keep source data separate from results
- `src/`: implementation
- `tests/`: focused automated tests

The initial inputs are example list C:

- `data/raw/Plant_traits_list_C.csv`
- `data/raw/optimization_report_C.txt`

The source files are retained in long-format/report form. Data
normalization belongs to Step 1 and must preserve duplicate trait intervals.

The Step 1 loader is in `src/plant_data.py`. It validates the CSV schema,
retains all source rows, reads quantities from the report's detailed plant
list, and does not silently fill missing numeric traits.

The Step 2 geometry implementation is in `src/geometry.py`. It creates the
rectangle, path, and hole masks on the 20 cm grid and computes padded
Euclidean boundary clearance maps for hard edge constraints.

The Step 3 count utilities are in `src/capacity.py`. They implement
largest-remainder allocation, the proposal's necessary disc-area screen, and
a bounded capacity-search helper. The area screen intentionally rejects
missing spacing values until a documented fallback is supplied.

The Step 4 rule model is in `src/rules.py`. It exposes hard-constraint
violations and separate penalties for grouping, height structure, local
flowering coverage, and structural-plant spacing.

The Step 5 constructor is in `src/greedy.py`. It supports both
most-constrained-first insertion and the matched random-order baseline, and
records energy evaluations and failed placements.

The Step 6 annealer is in `src/annealing.py`. It supports shift, swap, and
group proposals with seeded acceptance and cooling, while retaining the best
feasible plan. `energy_delta()` is currently an exact full-energy reference
used to verify a future spatially incremental implementation.

The Step 7 runner is in `src/experiments.py`. It records matched-seed
comparisons for constrained greedy, random-order greedy, and greedy plus
annealing, including feasibility, counts, energies, improvement, proportion
deviation, flowering coverage, runtime, and evaluation counts.

The shared scoring seam is in `src/evaluator.py`. It currently preserves the
exact full-energy reference while exposing a single delta API for replacing
full recomputation with local rule caches.

Immutable trait-derived values are identity-cached in `src/rules.py`, avoiding
repeated parsing during candidate scoring without changing penalty values.
Hard mask, edge, and pair-spacing checks are vectorized in the same module.
Height-rule neighborhood distances are also computed as one matrix per
candidate layout.

Spacing resolution is provided by `capacity.spacing_map()`. It requires an
explicit fallback for every missing spacing value; the current configuration
contains the confirmed Cichorium fallback only, so full experiments remain
blocked until the Epilobium value is confirmed.

For interim work, `pilot.prepare_exclusion_pilot()` supports an explicitly
labeled reduced pilot. It must receive the excluded taxon names and never
changes the source dataset.

## Configuration

The initial configuration is in `config/example_C.toml`. Paths are relative
to the repository root, so runs do not depend on the current working
directory.
