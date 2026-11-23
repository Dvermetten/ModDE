"""MkDocs hook (configured in mkdocs.yml): generate docs/parameters.md from the parameter registry.

The registry (src/modde/_registry.py) is the single source of truth (D-C3), so
this page is regenerated on every docs build and can never drift from the code.
"""
from __future__ import annotations

import re
from pathlib import Path

# Internal cross-references of the development notes (KI-27, D-D15, FR-10) mean
# nothing to readers of the documentation; they are removed from registry texts.
_ID = r"(?:(?:KI|FR)-\d+|D-[A-E]\d+)"
_ONLY_IDS = re.compile(rf"\s*\(\s*{_ID}(?:\s*[,;]\s*{_ID})*(?:\s+fixed)?\s*\)")
_TRAILING_IDS = re.compile(rf"\s*[,;]\s*{_ID}(?:\s*[,;]\s*{_ID})*")


def public(text: str) -> str:
    """Registry text without the development notes' issue / decision IDs."""
    return _TRAILING_IDS.sub("", _ONLY_IDS.sub("", text))

CATEGORIES = {
    "settings": "Problem and run",
    "population_size": "Population size",
    "initialization": "Initialisation",
    "mutation": "Mutation",
    "archive": "Archive",
    "crossover": "Crossover",
    "bound_correction": "Bound correction",
    "adaptation": "F and CR control",
    "oppositional_generation": "Oppositional generation",
}


def _ref_link(key: str, R, base: str = "") -> str:
    """Short label such as 'Tanabe & Fukunaga 2013' or 'Kononova et al. 2020a'.

    ``base`` is the page holding the reference list ("" = this page)."""
    first = R.REF[key].split(",")[0].strip()
    if "&" not in first and "et al" not in first:
        first += " et al."
    digits = "".join(ch for ch in key if ch.isdigit())
    suffix = key[len(key.rstrip("abcdefghijklmnopqrstuvwxyz")):] if key[-1].isalpha() else ""
    return f"[{first} {digits}{suffix}]({base}#ref-{key})"


def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _range(p, which: str) -> str:
    if which == "tune" and p.tune_doc:
        return p.tune_doc
    rng = p.tune if which == "tune" else p.valid
    if rng is None:
        return "–"
    if callable(rng):
        return "depends on d"
    lo, hi = rng
    text = f"[{_fmt(lo)}, {'∞' if hi is None else _fmt(hi)}]"
    if which == "tune" and p.log:
        text += ", log scale"
    return text


def _default(p) -> str:
    if p.default_doc:
        return p.default_doc
    if p.kind == "categorical":
        return f"`{p.options[0]}`"
    if p.default is None:
        return "–"
    return f"`{_fmt(p.default)}`"


def _active(p) -> str:
    if not p.active_when:
        return "always"
    return " **or** ".join(
        f"`{c.parent}` = " + " / ".join(f"`{v}`" for v in c.values) for c in p.active_when)


def param_section(p, R, base: str = "", level: str = "###") -> list[str]:
    """Documentation block of one parameter: heading, description, property table, options."""
    lines = ["", f"{level} `{p.name}`", "", public(p.doc), ""]
    rows = [("Type", p.kind), ("Default", _default(p))]
    if p.kind not in ("categorical", "bool"):
        rows.append(("Valid range", _range(p, "valid")))
    tunable = p.is_module and p.group is not None or p.tune is not None
    rows.append(("Tuning range", (("options" if p.kind in ("categorical", "bool")
                                   else _range(p, "tune")) if tunable else "not tuned")))
    rows.append(("Active when", _active(p)))
    if p.replaces:
        rows.append(("Replaces (0.0.4)", f"`{public(p.replaces)}`"))
    if p.references:
        rows.append(("References", "; ".join(_ref_link(r, R, base) for r in p.references)))
    lines += ["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    if p.kind == "categorical":
        lines += ["", "| Option | References |", "|---|---|"]
        for option in p.options:
            refs = R.OPTION_REFERENCES.get((p.name, option))
            text = "–" if refs is None else ("defined by modde" if not refs else
                                             "; ".join(_ref_link(r, R, base) for r in refs))
            default = " (default)" if option == p.options[0] else ""
            lines.append(f"| `{option}`{default} | {text} |")
    return lines


def render() -> str:
    from modde import _registry as R

    lines = [
        "# Parameters",
        "",
        "Every setting of modde, generated from the parameter registry "
        "(`modde._registry`). Settings are passed by name, for example "
        "`modde.fmin(f, budget=..., dim=..., lb=..., ub=..., mutation_base=\"TARGET\")`, "
        "or through `modde.settings_from_dict` / a ConfigSpace configuration.",
        "",
        "- **Defaults** give classic DE/rand/1/bin.",
        "- **Tunable** parameters are part of `modde.get_configspace(dim)` with the listed "
        "tuning range; parameters with an *active when* condition are conditional "
        "hyperparameters there.",
        "- A value you set yourself is never changed. If it conflicts with the modules, "
        "an error explains why; if it has no effect, you get a warning.",
        "",
        "## Overview",
        "",
        "| Parameter | Section | Default | Tuning range | Active when |",
        "|---|---|---|---|---|",
    ]
    for p in R.PARAMS:
        tunable = p.is_module and p.group is not None or p.tune is not None
        tune = ("options" if p.kind in ("categorical", "bool") else _range(p, "tune")) if tunable else "not tuned"
        lines.append(f"| [`{p.name}`](#{p.name.lower()}) | {CATEGORIES[p.category]} | {_default(p)} | "
                     f"{tune} | {_active(p)} |")

    for category, title in CATEGORIES.items():
        params = [p for p in R.PARAMS if p.category == category]
        lines += ["", f"## {title}"]
        for p in params:
            lines += param_section(p, R)

    lines += ["", "## Presets", "",
              "Named configurations, available as ConfigSpace configurations via "
              "`modde.get_presets(dim)` or by name (`preset=\"L-SHADE\"`).", "",
              "| Preset | Paper-exact | Reference | Notes |", "|---|---|---|---|"]
    for name, preset in R.PRESETS.items():
        refs = "; ".join(_ref_link(r, R) for r in preset.references)
        lines.append(f"| `{name}` | {'yes' if preset.exact else 'approximate'} | {refs} | "
                     f"{public(preset.doc)} {public(preset.notes)} |")

    lines += ["", "## References", ""]
    for key, text in R.REF.items():
        lines.append(f'- <a id="ref-{key}"></a>{text}')
    return "\n".join(lines) + "\n"


def on_pre_build(config, **kwargs):
    target = Path(config["docs_dir"]) / "parameters.md"
    content = render()
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    print(render())
