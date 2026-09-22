## Why

The current transaction-data benchmark improved substantially, but it is still only benchmark evidence. Before any serving promotion, the project needs a reproducible candidate package, feature/schema audit, and explicit promotion decision so model quality claims stay defensible.

## What Changes

- Add a non-serving model candidate export for the stronger transaction benchmark.
- Add feature/schema audit metadata alongside benchmark results.
- Run a larger bounded benchmark and record promotion-gate results.
- Keep the current serving artifacts unchanged until a later serving-compatibility change approves promotion.

## Capabilities

### New Capabilities

- `transaction-model-promotion`: Covers packaging, audit metadata, and promotion decision gates for transaction-data model candidates.

### Modified Capabilities

- `transaction-benchmark-modeling`: Clarify that strong benchmark outputs can produce candidate artifacts while still not automatically replacing serving artifacts.

## Impact

- Affected code: transaction benchmark adapter modules and benchmark tests.
- Affected artifacts: `artifacts/benchmark/transaction_data/**`.
- No production API contract changes.
- No raw dataset changes.
