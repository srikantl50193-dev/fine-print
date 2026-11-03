"""Download real policy documents into data/raw/ as <company>_<doctype>.md.

Sources are tried in order, and the first with a usable copy of a document wins:

1. OpenTermsArchive/contrib-versions: maintained and current, so preferred for "latest".
2. tosdr/tosdr-versions: named by the project brief. Its `main` branch is empty, because
   the data lives on `dataset-tosdr-*` tags, so we clone the newest of those.

If nothing can be downloaded at all, fictional samples are copied so the pipeline still runs.
"""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fine_print.config import get_settings

# Service name -> the folder names it may use in a collection.
SERVICES: dict[str, tuple[str, ...]] = {
    "spotify": ("Spotify",),
    "netflix": ("Netflix",),
    "uber": ("Uber",),
    "discord": ("Discord",),
    "reddit": ("Reddit",),
    "zoom": ("Zoom", "Zoom video communications"),
    "duolingo": ("Duolingo",),
    "airbnb": ("Airbnb",),
    "strava": ("Strava",),
    "pinterest": ("Pinterest",),
}
DOC_TYPES = {"terms": "Terms of Service.md", "privacy": "Privacy Policy.md"}
# Smaller files are consent banners or stubs, not real policies.
MIN_DOCUMENT_BYTES = 8_000


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    tag_prefix: str | None = None  # Clone the newest tag with this prefix, not the default branch.


SOURCES = (
    Source("contrib-versions", "https://github.com/OpenTermsArchive/contrib-versions"),
    Source(
        "tosdr-versions",
        "https://github.com/tosdr/tosdr-versions",
        tag_prefix="dataset-tosdr-",
    ),
)


def run_git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout


def latest_tag(url: str, prefix: str) -> str:
    """Find the newest tag. These tags end in ISO dates, so text order is date order."""
    output = run_git("ls-remote", "--tags", "--refs", url, f"refs/tags/{prefix}*")
    tags = sorted(line.rsplit("refs/tags/", 1)[1] for line in output.splitlines())
    if not tags:
        raise RuntimeError(f"No tags starting with {prefix} in {url}")
    return tags[-1]


def clone(source: Source, external_dir: Path) -> tuple[Path, str]:
    """Shallow-clone a source if needed, returning its path and the ref it holds."""
    target = external_dir / source.name
    ref = latest_tag(source.url, source.tag_prefix) if source.tag_prefix else "HEAD"
    if not target.exists():
        branch_args = ["--branch", ref] if source.tag_prefix else []
        run_git("clone", "--depth", "1", "--quiet", *branch_args, source.url, str(target))
    commit_date = run_git("-C", str(target), "log", "-1", "--format=%cs").strip()
    return target, f"{ref} ({commit_date})"


def find_document(root: Path, company: str, filename: str) -> Path | None:
    """Return the service's file, ignoring stubs too small to be a real policy."""
    for folder in SERVICES[company]:
        path = root / folder / filename
        if path.is_file() and path.stat().st_size >= MIN_DOCUMENT_BYTES:
            return path
    return None


def copy_documents(clones: dict[str, tuple[Path, str]], raw_dir: Path) -> dict[str, dict[str, str]]:
    """Copy each document from the first source that has it, and record its origin."""
    manifest: dict[str, dict[str, str]] = {}
    for company in SERVICES:
        for doc_type, filename in DOC_TYPES.items():
            for source_name, (root, ref) in clones.items():
                if path := find_document(root, company, filename):
                    target = raw_dir / f"{company}_{doc_type}.md"
                    shutil.copyfile(path, target)
                    manifest[target.name] = {
                        "source": source_name,
                        "ref": ref,
                        "path": str(path.relative_to(root)),
                    }
                    break
    return manifest


def print_report(manifest: dict[str, dict[str, str]]) -> None:
    for company in SERVICES:
        found = [
            f"{doc_type} [{manifest[f'{company}_{doc_type}.md']['source']}]"
            for doc_type in DOC_TYPES
            if f"{company}_{doc_type}.md" in manifest
        ]
        missing = [d for d in DOC_TYPES if f"{company}_{d}.md" not in manifest]
        status = "FOUND  " if not missing else ("PARTIAL" if found else "MISSING")
        detail = ", ".join(found) or "-"
        suffix = f"  (missing: {', '.join(missing)})" if missing else ""
        print(f"{status} {company:<10} {detail}{suffix}")


def use_samples(samples_dir: Path, raw_dir: Path) -> None:
    for sample in sorted(samples_dir.iterdir()):
        shutil.copyfile(sample, raw_dir / sample.name)
    print(f"WARNING: downloads failed, so FICTIONAL sample documents were copied to {raw_dir}.")


def main() -> None:
    settings = get_settings()
    settings.external_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    clones: dict[str, tuple[Path, str]] = {}
    for source in SOURCES:
        print(f"Fetching {source.name} ...")
        try:
            clones[source.name] = clone(source, settings.external_dir)
        except (subprocess.CalledProcessError, RuntimeError) as error:
            print(f"  failed: {error}", file=sys.stderr)
            continue
        root, ref = clones[source.name]
        folders = sum(path.is_dir() and path.name != ".git" for path in root.iterdir())
        print(f"  {ref}: {folders} service folders")

    manifest = copy_documents(clones, settings.raw_dir)
    if not manifest:
        use_samples(settings.samples_dir, settings.raw_dir)
        return

    print_report(manifest)
    (settings.raw_dir / "MANIFEST.json").write_text(
        json.dumps(
            {"retrieved": datetime.now(UTC).date().isoformat(), "documents": manifest},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n{len(manifest)} documents written to {settings.raw_dir}")


if __name__ == "__main__":
    main()
