"""MkDocs hook (configured in mkdocs.yml): generate docs/interactions.md from the
interaction study summary (docs/data/interactions_*.csv, produced by the
pairwise interaction study, dev/03_module_validation.md §5.3)."""
from __future__ import annotations

import csv
from pathlib import Path

CLASS_TEXT = {
    "incompatible": "the combination fails (error)",
    "no effect": "one of the two options has no effect in combination with the other (identical runs)",
    "conflict": "together they perform significantly worse than their separate effects predict",
    "synergy": "together they perform significantly better than their separate effects predict",
    "mixed": "significantly better on some problems and worse on others",
    "compatible": "no significant interaction: the effects simply add up",
}


def _read(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def render(data_dir: Path) -> str:
    single_path, pairs_path = data_dir / "interactions_single.csv", data_dir / "interactions_pairs.csv"
    lines = ["# Module interactions", ""]
    if not (single_path.exists() and pairs_path.exists()):
        return "\n".join(lines + ["*The interaction study results are not available in this build.*", ""])
    single, pairs = _read(single_path), _read(pairs_path)

    counts = {}
    for p in pairs:
        counts[p["cls"]] = counts.get(p["cls"], 0) + 1
    lines += [
        "How do module options behave **in combination**? Every non-default option was run alone and "
        "together with every option of every other module, against the classic-DE baseline.",
        "",
        "**Setup.** BBOB f1 (sphere), f8 (Rosenbrock), f15 (Rastrigin) and f21 (Gallagher), d = 5 and 20, "
        "10 runs each (5 instances × 2 seeds), budget 2000·d. The measure is anytime performance: the area "
        "under the ECDF of 51 precision targets (1e2 … 1e-8).",
        "",
        "**Interaction** of options A and B on one problem: "
        "`I = AUC(A+B) − AUC(A) − AUC(B) + AUC(baseline)`, i.e. how far the combination deviates from "
        "adding up the two separate effects. A cell is significant if |I| ≥ 0.02 and |z| > 3.29 "
        "(p ≈ 0.001, many comparisons). A pair is classified as:",
        "",
    ]
    for cls, text in CLASS_TEXT.items():
        n = counts.get(cls, 0)
        lines.append(f"- **{cls}** ({n} pair{'' if n == 1 else 's'}): {text}.")
    lines += ["", "Synergy and conflict require at least 2 of the 8 cells in the same direction.", "",
              "![Interaction matrix](img/interactions.png)", ""]

    lines += ["## Options on their own", "",
              "Each option alone versus classic DE (better/worse: significant in at least 2 of 8 cells, "
              "never in the other direction).", "",
              "| Option | Alone vs classic DE | Mean ΔAUC | Better cells | Worse cells |", "|---|---|---|---|---|"]
    for s in single:
        lines.append(f"| `{s['module']}={s['option']}` | {s['cls']} | {float(s['mean_delta_auc']):+.3f} | "
                     f"{s['better_cells']} | {s['worse_cells']} |")

    for cls in ("incompatible", "no effect", "conflict", "synergy", "mixed"):
        rows = [p for p in pairs if p["cls"] == cls]
        lines += ["", f"## {cls.capitalize()} ({len(rows)})", ""]
        if not rows:
            lines.append("None.")
            continue
        if cls == "no effect":
            # group by the option that has no effect
            groups: dict[str, list[str]] = {}
            for p in rows:
                silent = p["detail"].replace(" has no effect", "")
                partner = p["b"] if silent == p["a"] else p["a"]
                groups.setdefault(silent, []).append(partner)
            alone = {f"{s['module']}={s['option']}" for s in single if s["cls"] == "no effect"}
            lines += ["| Option without effect | Combined with | |", "|---|---|---|"]
            for silent, partners in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                note = ("also without effect on its own in classic DE; " if silent in alone else "") + \
                       f"{len(partners)} option{'' if len(partners) == 1 else 's'}"
                shown = ", ".join(f"`{x}`" for x in partners) if len(partners) <= 6 else "most other options"
                lines.append(f"| `{silent}` | {shown} | {note} |")
        elif cls == "incompatible":
            lines += ["| Option A | Option B | Detail |", "|---|---|---|"]
            lines += [f"| `{p['a']}` | `{p['b']}` | {p['detail']} |" for p in rows]
        else:
            rows.sort(key=lambda p: float(p["mean_interaction"]))
            lines += ["| Option A | Option B | Mean interaction | Range | Cells |", "|---|---|---|---|---|"]
            lines += [f"| `{p['a']}` | `{p['b']}` | {float(p['mean_interaction']):+.3f} | "
                      f"[{float(p['min_interaction']):+.3f}, {float(p['max_interaction']):+.3f}] | {p['detail']} |"
                      for p in rows]
    lines += ["", f"All other {counts.get('compatible', 0)} pairs are **compatible**.", ""]
    return "\n".join(lines)


def on_pre_build(config, **kwargs):
    docs = Path(config["docs_dir"])
    target = docs / "interactions.md"
    content = render(docs / "data")
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    print(render(Path(__file__).resolve().parents[1] / "docs" / "data"))
