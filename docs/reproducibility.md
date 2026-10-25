# Reproduction and custom replay

Catalog requests contain case, resolution_scale (1–3) and unsigned 32-bit seed. Refinement reduces physical integration time step or increases bench sample count. Config.dt_s is the maximum step; a shorter final interval preserves the exact requested duration.

Custom physical replay:
```sh
aeriallab flight-replay --config examples/custom-flight.json --output outputs/custom
aeriallab verify outputs/custom
```
Config rejects unknown fields and ambiguous boolean/timing inputs. Custom replay fingerprints input JSON bytes and freezes source provenance before dynamics run.

Catalog reproduction:
```sh
aeriallab batch --data-only --output outputs/catalog
aeriallab render outputs/catalog
aeriallab verify outputs/catalog
python examples/build_gallery.py outputs/catalog --destination docs/gallery
```
Captures retain source revision, dirty status, runtime versions, request SHA and resolved-parameter SHA. Numeric NPZ fields avoid pickle objects. CSV/JSON, plots and HTML have complete SHA-256 manifests.

Rendering does not rerun dynamics. Replay retains the original solve revision. Publication requires every default case from one clean committed source; later documentation and artifact commits can follow that recorded physical source.
