#!/usr/bin/env python3
"""Extend the newest DEMO500 summary with already-computed production artifacts.

Adds a ``bonus_artifacts/`` subfolder to the most recent ``*_demo500_summary``
directory, copying selected audit/plots/summary files from 6 earlier production
runs. No reruns. Preserves each source's audit/plots/summary structure.

Same principles as build_demo500_summary.py: copy (no symlink), log every cp.
If a whole source dir is MISSING -> log a warning and continue with the others.

Also updates the summary's inputs_manifest.json (new "bonus_artifacts" section)
and appends a section to README.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("build_demo500_bonus_artifacts")

# Each source: dest folder name, source run dir, and per-subdir glob patterns.
BONUS_SOURCES = [
    {
        "name": "phase_b4_edl_gate",
        "src": "outputs/runs/2026_05_27_112051_d_iqn_dss_phase_b4_edl_gate_production",
        "copy": [("audit", ["*.csv"]), ("plots", ["*.png"]), ("summary", ["*"])],
    },
    {
        "name": "phase_b5_ablation_original",
        "src": "outputs/runs/2026_05_27_181415_d_iqn_dss_phase_b5_ablation_suite",
        "copy": [("audit", ["*.csv"]), ("plots", ["*.png"]), ("summary", ["*"])],
    },
    {
        "name": "clean_25k_hold_diagnostic",
        "src": "outputs/runs/2026_05_24_204231_d_iqn_dss_clean_25k_hold_diagnostic_plots",
        "copy": [("plots", ["*.png"])],
    },
    {
        # Take everything incl. the large decision/transaction logs (thesis-critical).
        "name": "iqn_decision_audit",
        "src": "outputs/runs/2026_05_19_044638_d_iqn_dss_iqn_decision_audit_report",
        "copy": [("audit", ["*.csv"]), ("plots", ["*.png"]), ("summary", ["*"])],
    },
    {
        "name": "combined_iqn_hdp_audit",
        "src": "outputs/runs/2026_05_26_081515_d_iqn_dss_combined_iqn_hdp_audit_production",
        "copy": [("audit", ["*.csv"]), ("plots", ["*.png"]), ("summary", ["*"])],
    },
    {
        # No doubles: 10 plots from plots/, only the md/json metadata from summary/.
        "name": "iqn_vs_baseline_metrics",
        "src": "outputs/runs/2026_05_20_000840_d_iqn_dss_iqn_vs_baseline_metrics_plots",
        "copy": [("plots", ["*.png"]), ("summary", ["*.md", "*.json"])],
    },
]


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def resolve_summary_dir(cli_dir: str | None) -> Path:
    if cli_dir:
        d = Path(cli_dir) if Path(cli_dir).is_absolute() else ROOT / cli_dir
        if not d.exists():
            raise SystemExit(f"ERROR: --summary-dir not found: {d}")
        return d
    candidates = sorted((ROOT / "outputs" / "runs").glob("*_d_iqn_dss_demo500_summary"))
    if not candidates:
        raise SystemExit("ERROR: no *_demo500_summary dir found; pass --summary-dir")
    return candidates[-1]


def copy_one(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    size = src.stat().st_size
    log.info("cp %s  ->  %s", _rel(src), _rel(dst))
    return size


def build(summary_dir: Path, ts: str) -> None:
    bonus_root = summary_dir / "bonus_artifacts"
    bonus_root.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(summary_dir / "logs" / "bonus_artifacts_build.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    log.info("=== extending %s with bonus_artifacts ===", _rel(summary_dir))

    copied_sources: list[dict] = []
    missing_sources: list[str] = []

    for spec in BONUS_SOURCES:
        src_dir = ROOT / spec["src"]
        dest_dir = bonus_root / spec["name"]
        if not src_dir.exists():
            log.warning("MISSING source, skipping: %s", spec["src"])
            missing_sources.append(spec["src"])
            continue

        n_files = 0
        n_bytes = 0
        for subdir, patterns in spec["copy"]:
            src_sub = src_dir / subdir
            if not src_sub.exists():
                log.warning("  no %s/ in %s", subdir, spec["name"])
                continue
            seen: set[Path] = set()
            for pat in patterns:
                for f in sorted(src_sub.glob(pat)):
                    if not f.is_file() or f in seen:
                        continue
                    seen.add(f)
                    n_bytes += copy_one(f, dest_dir / subdir / f.name)
                    n_files += 1
            if not seen:
                log.warning("  %s/%s matched 0 files", spec["name"], subdir)

        log.info("source %s: %d files, %.1f MB", spec["name"], n_files, n_bytes / 1e6)
        copied_sources.append(
            {
                "name": spec["name"],
                "path": spec["src"],
                "mtime": _mtime(src_dir),
                "n_files": n_files,
                "bytes": n_bytes,
            }
        )

    total_bytes = sum(s["bytes"] for s in copied_sources)
    total_files = sum(s["n_files"] for s in copied_sources)
    log.info("bonus_artifacts total: %d files, %.1f MB across %d sources",
             total_files, total_bytes / 1e6, len(copied_sources))

    _update_manifest(summary_dir, ts, copied_sources, missing_sources, total_bytes)
    _update_readme(summary_dir, ts, copied_sources, missing_sources)
    log.info("=== done ===")


def _update_manifest(summary_dir: Path, ts: str, sources: list[dict],
                     missing: list[str], total_bytes: int) -> None:
    path = summary_dir / "inputs_manifest.json"
    if not path.exists():
        raise SystemExit(f"ERROR: manifest not found: {path}")
    manifest = json.loads(path.read_text())
    manifest["bonus_artifacts"] = {
        "added_at": ts,
        "n_sources": len(sources),
        "total_bytes": total_bytes,
        "sources": sources,
        "missing_sources": missing,
    }
    path.write_text(json.dumps(manifest, indent=2))
    log.info("updated %s (bonus_artifacts section, %d sources)", _rel(path), len(sources))


def _update_readme(summary_dir: Path, ts: str, sources: list[dict],
                   missing: list[str]) -> None:
    path = summary_dir / "README.md"
    names = {s["name"] for s in sources}
    descriptions = {
        "phase_b4_edl_gate": "EDL gate production: audit CSVs, 6 plots, summary.",
        "phase_b5_ablation_original": "original B.5 ablation suite: audit CSVs, 9 plots.",
        "clean_25k_hold_diagnostic": "10 plots documenting the IQN HOLD-bias.",
        "iqn_decision_audit": "full DSS decision audit: decision log (by_step), "
                              "transaction log (trades_only), structured summaries, heatmap.",
        "combined_iqn_hdp_audit": "combined IQN+HDP audit CSVs + plot.",
        "iqn_vs_baseline_metrics": "10 IQN-vs-baseline metric plots + md/json summary.",
    }
    lines = [
        "",
        "## Bonus artifacts (bonus_artifacts/)",
        "",
        f"Already-computed production artifacts from earlier pipeline phases "
        f"(added {ts}). No reruns. Each subfolder preserves its source "
        f"audit/plots/summary structure.",
        "",
    ]
    for spec in BONUS_SOURCES:
        if spec["name"] in names:
            lines.append(f"- `{spec['name']}/` — {descriptions.get(spec['name'], '')}")
    if missing:
        lines.append("")
        lines.append(f"_Missing sources (skipped): {', '.join(missing)}_")
    lines.append("")
    with path.open("a") as fh:
        fh.write("\n".join(lines))
    log.info("appended bonus_artifacts section to %s", _rel(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-dir", default=None,
                    help="demo500_summary dir (default: newest *_demo500_summary).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    summary_dir = resolve_summary_dir(args.summary_dir)
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    build(summary_dir, ts)
    print(f"\n[build_demo500_bonus_artifacts] EXTENDED: {summary_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
