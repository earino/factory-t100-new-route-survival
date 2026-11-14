---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

# Will a newly launched U.S. domestic nonstop airline route still be operating a year later? Forward route-survival prediction from T-100 segment reports

When a U.S. certificated carrier starts scheduled passenger nonstop service on a domestic city market pair it has not served in the previous 12 months, will that service still be operating in the anniversary quarter 12-14 months later?

Derived from U.S. Department of Transportation, Bureau of Transportation Statistics (Office of Airline Information), distributed via TranStats. This is not its official product.

Prediction time: Ninety days after the end of entry month M, after the monthly T-100 compilation for M is assumed available. Only T-100 records for months at or before M may enter features; months after M are unseen.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.6478237991430504,
    "brier": 0.2677095943981317
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.2106785523522104
  },
  "calendar_identity": {
    "auc": 0.5019817812796841,
    "brier": 0.21298999804210061
  }
}
```

## Limitations

- No licence statement appears on the TranStats download page itself; the public-domain designation is taken from BTS's own data-portal metadata for T-100-derived products plus 17 U.S.C. 105, with BTS requiring source citation. Anyone re-publishing should carry the BTS credit line rather than assert a CC0 dedication over the upstream data.
- The download path is an ASP.NET form postback with view-state tokens, not a stable static file URL; eleven annual pulls must be scripted, and the interface could change. The 'Prezipped File' option is the documented fallback.
- The label treats a seasonal suspension as an exit whenever the carrier performs no departures in the three-month anniversary window, so genuinely seasonal routes launched in the wrong season are systematically coded as failures.
- Corporate events are confounded with route failure: mergers, certificate transfers and UniqueCarrier re-coding make an acquired carrier's routes vanish from the panel and be labelled discontinued even where service continued under another code.
- Entry episodes are not independent within carrier or within airport: a carrier that shuts down fails all of its new routes at once, and a hub's entry cohort shares the same local demand shock. Row counts overstate the effective number of independent events.
- Domestic U.S. certificated carriers only: foreign-carrier entries, non-scheduled/charter service (classes L/N/P/Q/R/V) and all-cargo-only service are excluded, so results do not describe the whole market.
- T-100 coverage of very small commuter operators is uneven and reporting practices vary, which is partly why the >=8 departures-in-entry-month rule is used; that rule also removes genuine but very thin new services.
- City-market aggregation deliberately hides airport-level shifts within a metro (e.g. a move from one metro airport to another counts as continuous service), which is a definitional choice, not a measurement of airport-level survival.
- All entries after 2018-12 are excluded because their anniversary windows cross the 2020 schedule collapse, so the task describes pre-2019 market structure and is a retrospective, current-vintage construction rather than proof of what an operator could have computed live in 2013.
- No performance metrics were computed and none should be inferred from this selection; the base rate and the entry counts above are estimates pending construction.
- No historical vintages or full-row availability ledger are supplied; the current TranStats snapshot is used and training_label_timing is unverified.
- No publisher publication schedule was supplied; a conservative 90-day administrative availability lag is assumed for timing metadata and is not a measured publication date.
- The source has month-level dates only; the qualification event cluster is the carrier identity, not a day-level event timestamp.
- T-100 publication lag is not documented in the supplied evidence; no zero-day publication date is invented.
- Absence of a T-100 pair-month row is treated as zero performed departures only after global month coverage and missing-DepPerformed checks; missing feature values are blank rather than zero.
- City-market aggregation hides airport-level shifts inside a metro.
- Seasonal suspensions, mergers, certificate transfers and carrier re-coding can be coded as route exits.
- Entries are not independent within a carrier or hub; route counts overstate independent events.
- The frozen population is U.S. certificated carriers and Class F scheduled service; foreign carriers, non-scheduled classes and all-cargo-only service are not described.
- Small commuter coverage is uneven and the at-least-8-departures rule removes very thin services.
- Entries after 2018-12 are excluded because their anniversary windows cross the 2020 schedule collapse.
- The label uses all-class DepPerformed over M+12 through M+14 as frozen; if only scheduled passenger service is intended, the semantics differ.
- The base rate is measured by construction and is not assumed.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
