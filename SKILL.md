# Architectural Invariants & Worker Rules

1. **Scope Boundary**: Touch ONLY files specified in your ticket `allowed_files`. Any modification outside this set triggers instant failure.
2. **Eval-Driven Verification**: A ticket is strictly COMPLETE when the specified `eval` command exits with code `0`.
3. **Structured Telemetry**: On failure or test rejection, report ONLY `{ exit_code, error_type, failed_assertion }`. Never dump raw terminal traces.
4. **No Direct Feature Code from Orchestrator**: Workers own implementation and tests within ticket bounds.
5. **Clean Code & Typings**: Use strict typing, modular decoupled units, and minimal dependencies.
6. **Frontend Token Compliance**: Zero arbitrary px or raw hex values; use tokens defined in `DESIGN.md`.
