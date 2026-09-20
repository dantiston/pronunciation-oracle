## Agent Working Principles

These principles are binding. Follow them for every change unless explicitly told otherwise.

### Core Workflow

**1. MUST use Test-Driven Development**
Write a failing test first, then the minimal implementation to pass it, then refactor.
Keep tests fast, deterministic, and isolated. Run the relevant suite before marking work done. Never submit with failing or skipped tests.

**2. MUST ensure unit test correctness**
All unit tests must be correct. Do not write a unit test that is incorrect.
If you cannot verify the correctness of a unit test, do not write the unit test and ask
for clarification. Do not write unit tests that are vague or underspecified, such as
checking that an integer is less than or equal to a value when it should be a specific
value. Utilize existing, tested equality operators as much as possible, such as
checking equality between two entire data structures instead of picking and choosing
members.

**3. MUST make small, reversible changes**
One logical change per change. Prefer editing existing code over adding new abstractions. Do not refactor unrelated code.

### Code Quality

**4. MUST use strict typing**
All functions, methods, and public APIs require full type annotations. No implicit `Any`. Prefer `Protocol`, `TypedDict`, generics, and explicit return types. Code must pass `ty`. Prefer using builtins directly (e.g. `dict`) over
typing imports (e.g. `Dict`).

**5. SHOULD follow YAGNI, KISS, DRY**
Build only what is asked for. Prefer simple, readable solutions over clever ones. Extract shared logic instead of duplicating, but do not over-abstract prematurely.

**6. MUST follow existing patterns**
Match the surrounding code style, naming, file layout, and error handling.
Run `pytest`, `ruff check`, `ruff format`, and `ty check` before finishing.

### Performance & Correctness

**7. MUST ensure reproducibility**
Seed all randomness. Avoid nondeterministic ordering. Document any nondeterminism you cannot remove.

**8. MUST fail loudly**
No silent `except: pass`, no swallowed errors, no default fallbacks that hide bugs. Validate inputs early and raise informative errors. Log key decisions at appropriate levels.

### Safety & Hygiene

**9. MUST NOT introduce secrets or side effects**
No API keys, tokens, or absolute local paths in code. No network calls, file writes, or global state mutation unless explicitly requested.

**10. MUST document public behavior**
Every public function/class needs a docstring: what it does, args, returns, and edge cases. Update docs when behavior changes.
