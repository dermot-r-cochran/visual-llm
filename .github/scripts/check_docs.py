"""Doc check for visual-llm's Markdown (standard library only).

Four checks, on the Markdown sources:

1. Every relative link in README.md and under docs/ resolves to a file or
   directory that exists. External links and in-page anchors are not checked.
2. No Markdown file carries a second front-matter block: a `---` line, then
   only `key: value` lines, then `---`, anywhere but at the very top. That
   is what a stray fragment left by a merge looks like.
3. Every test the README cites (`tests/<file>.py::<name>`, or `::<name>`
   after one) exists: each name in the chain is a function or class in that
   file. A README must never cite a test that is not there.
4. What the README counts or tabulates matches the code: "Four levels" and
   the levels table against the `l0`..`l3` fields of `HIDLAnnotation` in
   hidl/schema.py, the fields the L0 and L3 rows name against those
   dataclasses, "Three modes" and the modes table against `AnnotationMode` in
   hidl/modes.py, and "the core library has none" (no runtime dependency)
   with `openai` as an optional extra, against pyproject.toml.

The shape follows architecture-definition-model's .github/scripts/check_docs.py.
The README-proof convention it backs was Dermot's decision of 10 October
2026: every capability row or bullet in the README names the test that proves
it, or says "no test yet" or "not yet implemented".

Run from anywhere: python .github/scripts/check_docs.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {"node_modules", "build", "dist"}
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?")
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
KEY = re.compile(r"^[A-Za-z_][\w-]*\s*:(\s|$)")
FENCE = re.compile(r"^\s*(```|~~~)")
CITE = re.compile(r"`(tests/[\w/.-]+\.py)?((?:::\w+)+)`")
NUMBERS = {w: i for i, w in enumerate([
    "zero", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve",
])}


def markdown_files() -> list[Path]:
    found = []
    for path in sorted(ROOT.rglob("*.md")):
        parts = path.relative_to(ROOT).parts
        if any(p.startswith(".") or p in SKIP or p.endswith(".egg-info") for p in parts[:-1]):
            continue
        found.append(path)
    return found


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def prose(lines: list[str]) -> list[tuple[int, str]]:
    """Lines outside fenced code blocks, with their line numbers."""
    kept, fenced = [], False
    for number, line in enumerate(lines, start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            kept.append((number, line))
    return kept


def front_matter(path: Path, errors: list[str]) -> None:
    lines, fenced, i = read(path), False, 0
    while i < len(lines):
        if FENCE.match(lines[i]):
            fenced = not fenced
        elif not fenced and lines[i].strip() == "---":
            j = i + 1
            while j < len(lines) and KEY.match(lines[j]):
                j += 1
            if j > i + 1 and j < len(lines) and lines[j].strip() == "---":
                if i > 0:
                    errors.append(f"{rel(path)}:{i + 1}: a front-matter block below the top")
                i = j
        i += 1


def links(path: Path, errors: list[str]) -> None:
    for number, line in prose(read(path)):
        line = INLINE_CODE.sub("", line)
        targets = [m.group(1) for m in LINK.finditer(line)]
        ref = REF.match(line)
        if ref:
            targets.append(ref.group(1))
        for target in targets:
            if SCHEME.match(target) or target.startswith("#"):
                continue
            bare = target.split("#", 1)[0].split("?", 1)[0]
            if not bare:
                continue
            base = ROOT if bare.startswith("/") else path.parent
            if not (base / bare.lstrip("/")).exists():
                errors.append(f"{rel(path)}:{number}: link {target!r} resolves to nothing")


def defined(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return {node.name for node in ast.walk(tree) if isinstance(node, kinds)}


def citations(path: Path, errors: list[str]) -> int:
    seen, current = 0, None
    for number, line in prose(read(path)):
        for match in CITE.finditer(line):
            current = match.group(1) or current
            names = match.group(2).strip(":").split("::")
            where = f"{rel(path)}:{number}: cites {match.group(0)}"
            if current is None:
                errors.append(f"{where} with no test file before it")
                continue
            target = ROOT / current
            if not target.is_file():
                errors.append(f"{where}, and {current} does not exist")
                continue
            missing = [n for n in names if n not in defined(target)]
            if missing:
                errors.append(f"{where}, and {current} defines no {', '.join(missing)}")
            seen += 1
    return seen


def number(word: str) -> int | None:
    return int(word) if word.isdigit() else NUMBERS.get(word.lower())


def table_after(lines: list[str], header: str) -> list[list[str]]:
    """The body rows of the first table whose header row starts with `header`."""
    rows: list[list[str]] = []
    for i, line in enumerate(lines):
        if line.startswith(header):
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                rows.append([cell.strip() for cell in row.strip().strip("|").split("|")])
            break
    return rows


def class_body(tree: ast.Module, name: str) -> list[ast.stmt]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node.body
    return []


def fields(tree: ast.Module, name: str) -> list[str]:
    """The annotated field names of a dataclass, in order."""
    return [n.target.id for n in class_body(tree, name)
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]


def code(errors: list[str]) -> None:
    lines = read(ROOT / "README.md")
    text = "\n".join(lines)
    schema = ast.parse((ROOT / "hidl" / "schema.py").read_text(encoding="utf-8"))

    levels = [f.upper() for f in fields(schema, "HIDLAnnotation") if re.fullmatch(r"l\d", f)]
    rows = table_after(lines, "| Level |")
    said = re.search(r"^## (\w+) levels$", text, re.MULTILINE)
    if not said or number(said.group(1)) != len(levels):
        errors.append(f"README.md: the level count does not match HIDLAnnotation's "
                      f"{len(levels)} levels")
    if [row[0].strip("*") for row in rows] != levels:
        errors.append(f"README.md: the levels table lists {[row[0] for row in rows]}, "
                      f"and HIDLAnnotation has {levels}")
    for row in rows:
        level = row[0].strip("*")
        named = re.findall(r"`(\w+)`", row[1])
        if named and named != fields(schema, level):
            errors.append(f"README.md: the {level} row names {named}, "
                          f"and {level} has {fields(schema, level)}")

    modes = ast.parse((ROOT / "hidl" / "modes.py").read_text(encoding="utf-8"))
    values = [n.value.value for n in class_body(modes, "AnnotationMode")
              if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)]
    rows = table_after(lines, "| Mode |")
    said = re.search(r"^## (\w+) modes$", text, re.MULTILINE)
    if not said or number(said.group(1)) != len(values):
        errors.append(f"README.md: the mode count does not match AnnotationMode's "
                      f"{len(values)} values")
    tabled = [re.sub(r"`([^`]+)`.*", r"\1", row[0]) for row in rows]
    if tabled != values:
        errors.append(f"README.md: the modes table lists {tabled}, "
                      f"and AnnotationMode has {values}")

    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if "the core library has\nnone" in text or "the core library has none" in text:
        if not re.search(r"^dependencies\s*=\s*\[\s*\]", project, re.MULTILINE):
            errors.append("README.md: says the core library has no dependency, "
                          "and pyproject.toml declares one")
        if not re.search(r"^openai\s*=", project, re.MULTILINE):
            errors.append("README.md: says openai is an optional dependency, "
                          "and pyproject.toml has no openai extra")


def main() -> int:
    errors: list[str] = []
    files = markdown_files()
    checked = [p for p in files if rel(p) == "README.md" or rel(p).startswith("docs/")]
    for path in files:
        front_matter(path, errors)
    for path in checked:
        links(path, errors)
    cited = citations(ROOT / "README.md", errors)
    code(errors)
    for error in errors:
        print(error)
    print(f"{len(files)} Markdown files, {len(checked)} link-checked, "
          f"{cited} test citations, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
