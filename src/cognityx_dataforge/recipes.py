from __future__ import annotations

import warnings


RECIPE_ALIASES = {"v0": "paragraph-qa", "v1": "knowledge-unit-qa"}
SUPPORTED_RECIPES = frozenset({"paragraph-qa", "knowledge-unit-qa"})


def normalize_recipe(recipe: str | None = None, *, variant: str | None = None) -> str:
    selected = recipe or variant
    if not selected:
        raise ValueError("A recipe is required")
    if variant is not None:
        warnings.warn("--variant is deprecated; use --recipe", DeprecationWarning, stacklevel=2)
    canonical = RECIPE_ALIASES.get(selected, selected)
    if canonical not in SUPPORTED_RECIPES:
        raise ValueError(f"Unsupported DataForge recipe: {selected}")
    return canonical
