# AI/ML Auditor Skill and FraudGuard Hardening Design

## Objective

Install the supplied `ai-ml-project-auditor-fixer` as a reusable global Codex
skill, then use it through an OpenSpec-governed change to audit, repair, and
verify the FraudGuard repository end to end.

## Skill Packaging

The supplied single-file skill will be converted into a progressive-disclosure
package rather than installed verbatim. Its activation description and core
decision rules will remain in `SKILL.md`; detailed audit checklists,
project-archetype branches, engineering checks, and the required final-report
contract will move into focused references. Mojibake will be corrected without
changing the intended requirements.

The global package will contain:

- `SKILL.md` for activation, authority boundaries, audit phases, prioritization,
  and reference routing.
- `agents/openai.yaml` for discoverable UI metadata.
- `references/ml-audit.md` for problem, data, leakage, preprocessing, modeling,
  tuning, metrics, threshold, and calibration checks.
- `references/project-branches.md` for fraud, forecasting, deep learning, RAG,
  LLM, agentic, and fine-tuning concerns.
- `references/engineering-audit.md` for software, inference, deployment,
  monitoring, reproducibility, documentation, verification, and regression
  protection.
- `references/report-contract.md` for severities, fix dispositions, completion
  criteria, and the required nine-section final report.

The package will be checked with the official skill validator before use.

## OpenSpec Workflow

OpenSpec will be initialized for Codex in this repository. A change named
`audit-and-harden-fraudguard` will capture the proposal, technical design,
behavioral requirements, and an ordered task list. The change will be strictly
validated before implementation begins and again after tasks are completed.

The executable plan will use correctness gates in this order:

1. Establish the real objective, label semantics, unit of prediction, and
   production-time feature availability.
2. Audit the dataset, split, duplicate/entity overlap, temporal assumptions,
   and leakage risks.
3. Audit fit/transform boundaries, categorical handling, resampling, and
   training-serving parity.
4. Audit baseline/model selection, tuning isolation, threshold selection,
   calibration, and imbalanced-class metrics.
5. Trace one request through API validation, preprocessing artifacts, model,
   thresholding, response semantics, and failure behavior.
6. Repair critical and high-severity defects before maintainability or
   documentation work.
7. Add regression tests for corrected critical behavior.
8. Run proportionate tests, DVC stage checks, and smoke evaluations. Expensive
   full training will only run when feasible; otherwise the report will clearly
   separate executed verification from remaining full-run work.
9. Align documentation and project claims with reproducible evidence.

## Change Boundaries

Raw source data will not be destructively modified. Existing public interfaces
will be preserved unless they are technically invalid. Unrelated rewrites and
unjustified platform complexity are out of scope. No metric or claim will be
reported as verified unless it was reproduced or directly traced to reliable
evidence.

Material ambiguity that changes the target, positive class, prediction time,
or business cost will be raised explicitly. Low-risk implementation details
will be inferred from the repository and recorded as assumptions.

## Verification and Completion

Completion requires a validated global skill, a valid OpenSpec change, repaired
critical/high defects where feasible, regression coverage for important fixes,
executed checks with honest pass/fail/not-run separation, coherent training and
inference behavior, and a final report following the skill's nine-section
contract.

The OpenSpec change will only be archived after its achievable tasks pass and
remaining blockers are explicitly documented.
