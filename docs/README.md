# Project living documentation

This folder is the durable source of truth for the restoration project. The documents are deliberately kept in the repository so their history is reviewable with the same `feature → develop → main` workflow as the code.

## Documents

- `PLAN.md` — delivery phases, acceptance criteria, risks, and current next work.
- `MEMORY.md` — facts, decisions, validation evidence, and a dated change log.
- `architecture.md` — component and data architecture.
- `feature-map.md` — implemented capabilities, ownership, and planned extensions.
- `request-flow.md` — inference request lifecycle and failure paths.

## Update protocol

At the end of each meaningful implementation, experiment, or release change:

1. Update `MEMORY.md` with the decision, evidence, and date.
2. Update `PLAN.md` if scope, status, risk, or acceptance criteria changed.
3. Update one or more diagrams when a component, capability, or request path changed.
4. Include the documentation changes in the same feature branch and promotion as the related code.

Do not overwrite historical metrics or decisions; append a dated entry and correct the current-state section when needed.
