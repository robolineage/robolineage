"""Shared fixture pack for cross-domain contract tests.

Contracts/runtime tooling is the only writer; other domains use these
fixtures verbatim in integration tests so cross-domain contracts are exercised
against a single source of truth. See `README.md` for inventory + freeze rules.
"""
