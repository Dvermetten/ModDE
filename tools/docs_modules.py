"""MkDocs hook (configured in mkdocs.yml): fill the generated parts of the module
reference pages (docs/modules/*.md) and the presets page.

The pages are written by hand (what a module does, formulas, examples). They
contain markers that this hook replaces when the page is rendered:

    <!-- modde:params CATEGORY -->         parameter blocks of a registry category
    <!-- modde:interactions CATEGORY -->   interaction-study results of its options
    <!-- modde:presets -->                 every preset with its explicit settings

The registry (src/modde/_registry.py) and the interaction study summary
(docs/data/interactions_*.csv) are the sources, so the tables cannot drift
from the code.
"""
from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_modde_docs_parameters", _HERE / "docs_parameters.py")
_params = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_params)

MARKER = re.compile(r"<!--\s*modde:(\w+)\s*([\w-]*)\s*-->")
PRESET_DIMS = (10, 30)

CLASS_ORDER = ("incompatible", "no effect", "conflict", "synergy", "mixed")


def _base(page) -> str:
    """Relative link from ``page`` to parameters.md (which holds the reference list)."""
    depth = page.file.src_uri.count("/")
    return "../" * depth + "parameters.md"


def _params_block(category: str, R, base: str) -> str:
    lines: list[str] = []
    for p in R.PARAMS:
        if p.category == category:
            lines += _params.param_section(p, R, base, level="###")
    return "\n".join(lines) + "\n"


def _read(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _interactions_block(category: str, R, data: Path) -> str:
    names = {p.name for p in R.PARAMS if p.category == category}
    single = [s for s in _read(data / "interactions_single.csv") if s["module"] in names]
    pairs = _read(data / "interactions_pairs.csv")
    if not single and not pairs:
        return "*The interaction study results are not available in this build.*\n"
    if not single:
        return ("These options were not part of the interaction study "
                "(see [Module interactions](../interactions.md)).\n")
    mine = lambda opt: opt.split("=")[0] in names  # noqa: E731
    lines = ["From the [interaction study](../interactions.md) (BBOB, d = 5 and 20, "
             "anytime performance against classic DE).", "",
             "| Option | Alone vs classic DE | Mean ΔAUC |", "|---|---|---|"]
    for s in single:
        lines.append(f"| `{s['module']}={s['option']}` | {s['cls']} | {float(s['mean_delta_auc']):+.3f} |")
    silent_alone = {f"{s['module']}={s['option']}" for s in single if s["cls"] == "no effect"}
    silent_of = lambda p: p["detail"].replace(" has no effect", "")  # noqa: E731

    def shown(p) -> bool:
        if p["cls"] not in CLASS_ORDER or not (mine(p["a"]) or mine(p["b"])):
            return False
        if p["cls"] != "no effect":
            return True
        # a "no effect" pair is shown on the page of the option without effect, and
        # only if that option does have an effect in classic DE
        return mine(silent_of(p)) and silent_of(p) not in silent_alone

    notable = sorted((p for p in pairs if shown(p)),
                     key=lambda p: (CLASS_ORDER.index(p["cls"]), p["a"], p["b"]))
    lines += [""]
    for option in sorted(silent_alone):
        lines += [f"`{option}` has no effect in classic DE: it only acts together with other "
                  "modules (see its description above).", ""]
    if not notable:
        lines.append("All other combinations with options of other modules are **compatible** "
                     "(the effects add up).")
    else:
        lines += ["Combinations that are not simply compatible:", "",
                  "| Option | Combined with | Class | Detail |", "|---|---|---|---|"]
        for p in notable:
            if p["cls"] == "no effect":
                a = silent_of(p)
                b = p["b"] if a == p["a"] else p["a"]
                detail = f"`{a}` has no effect"
            else:
                a, b = (p["a"], p["b"]) if mine(p["a"]) else (p["b"], p["a"])
                detail = p["detail"]
                if p["cls"] != "incompatible":
                    detail = f"mean interaction {float(p['mean_interaction']):+.3f}; {detail}"
            lines.append(f"| `{a}` | `{b}` | {p['cls']} | {detail} |")
        lines += ["", "All other combinations are compatible."]
    return "\n".join(lines) + "\n"


def _fmt_value(v) -> str:
    if isinstance(v, bool):
        return f"`{v}`"
    if isinstance(v, str):
        return f'`"{v}"`'
    if isinstance(v, float):
        return f"`{v:g}`"
    return f"`{v}`"


def _presets_block(R, base: str) -> str:
    lines: list[str] = []
    d0, d1 = PRESET_DIMS
    for name, preset in R.PRESETS.items():
        s0, s1 = preset.settings(d0), preset.settings(d1)
        refs = "; ".join(_params._ref_link(r, R, base) for r in preset.references)
        lines += ["", f"## `{name}`", "", _params.public(preset.doc) + ("" if preset.exact else " **Approximation.**"), ""]
        if refs:
            lines += [f"Reference: {refs}.", ""]
        if preset.notes:
            lines += [f"!!! note \"Notes\"\n    {_params.public(preset.notes)}", ""]
        if not s0:
            lines += ["No settings: all defaults.", ""]
            continue
        lines += ["| Setting | Value |", "|---|---|"]
        for key, value in s0.items():
            shown = _fmt_value(value)
            if s1.get(key) != value:
                shown += f" at d = {d0} ({_fmt_value(s1.get(key))} at d = {d1}; depends on d)"
            lines.append(f"| [`{key}`]({base}#{key.lower()}) | {shown} |")
        lines += [""]
    return "\n".join(lines) + "\n"


def on_page_markdown(markdown: str, page, config, files, **kwargs):
    if "modde:" not in markdown:
        return markdown
    from modde import _registry as R

    data = Path(config["docs_dir"]) / "data"
    base = _base(page)

    def replace(match: re.Match) -> str:
        kind, arg = match.group(1), match.group(2)
        if kind == "params":
            if not any(p.category == arg for p in R.PARAMS):
                raise ValueError(f"{page.file.src_uri}: unknown registry category {arg!r}")
            return _params_block(arg, R, base)
        if kind == "interactions":
            return _interactions_block(arg, R, data)
        if kind == "presets":
            return _presets_block(R, base)
        raise ValueError(f"{page.file.src_uri}: unknown marker modde:{kind}")

    return MARKER.sub(replace, markdown)
