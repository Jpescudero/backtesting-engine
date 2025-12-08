from __future__ import annotations

import os
import sys
from pathlib import Path
from trace import _find_executable_linenos, Trace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
COVERAGE_DIR = PROJECT_ROOT / "coverage-report"
DEFAULT_THRESHOLD = float(os.environ.get("COVERAGE_THRESHOLD", "80"))


def _is_project_file(path: Path) -> bool:
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return path.is_relative_to(SRC_DIR)


def _gather_coverage(results) -> dict[Path, dict[str, float]]:
    executed: dict[Path, set[int]] = {}
    for filename, lineno in results.counts:
        path = Path(filename).resolve()
        if not _is_project_file(path):
            continue
        executed.setdefault(path, set()).add(lineno)

    coverage_map: dict[Path, dict[str, float]] = {}
    for path, executed_lines in executed.items():
        executable = set(_find_executable_linenos(str(path)))
        total = len(executable)
        if total == 0:
            continue
        covered = len(executable & executed_lines)
        percent = (covered / total) * 100
        coverage_map[path] = {
            "covered": covered,
            "total": total,
            "percent": percent,
        }
    return coverage_map


def _write_report(coverage_map: dict[Path, dict[str, float]]) -> float:
    COVERAGE_DIR.mkdir(exist_ok=True)
    lines = ["Archivo,Lineas cubiertas,Lineas totales,Cobertura (%)"]
    total_covered = 0
    total_lines = 0
    for path in sorted(coverage_map):
        stats = coverage_map[path]
        total_covered += int(stats["covered"])
        total_lines += int(stats["total"])
        lines.append(f"{path.relative_to(PROJECT_ROOT)},{stats['covered']},{stats['total']},{stats['percent']:.2f}")

    overall = (total_covered / total_lines * 100) if total_lines else 0.0
    lines.append(f"TOTAL,{total_covered},{total_lines},{overall:.2f}")

    report_path = COVERAGE_DIR / "coverage.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reporte de cobertura escrito en {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Cobertura total: {overall:.2f}%")
    return overall


def main() -> int:
    tracer = Trace(count=True, trace=False, ignoredirs=[sys.prefix, sys.exec_prefix])
    exit_code = tracer.runfunc(pytest.main, ["tests"])
    results = tracer.results()

    if exit_code != 0:
        return int(exit_code)

    coverage_map = _gather_coverage(results)
    overall = _write_report(coverage_map)

    threshold = DEFAULT_THRESHOLD
    if overall < threshold:
        print(
            f"Cobertura ({overall:.2f}%) por debajo del umbral requerido ({threshold:.0f}%).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
