"""Download real policy documents into data/raw/ as <company>_<doctype>.md.

`DOCUMENTS` names the exact source, folder and file of every document, so nothing is guessed:

1. OpenTermsArchive/contrib-versions: maintained and current, so used wherever it has a real copy.
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

CONTRIB = "contrib-versions"
TOSDR = "tosdr-versions"
DOC_TYPES = ("terms", "privacy")
# Smaller files are consent banners or stubs, not real policies.
MIN_DOCUMENT_BYTES = 8_000


@dataclass(frozen=True)
class DocumentLocation:
    """Where one document lives: its source, and its exact folder and file in that source."""

    source: str
    folder: str
    filename: str


# Neither source has Uber or Duolingo. Airbnb's only privacy document is a captured login
# wall (byte-identical to its "Trackers Policy"). Lyft, Coursera and PayPal take their places.
# Strava is only in tosdr-versions, whose newest snapshot is older than contrib-versions.
DOCUMENTS: dict[str, dict[str, DocumentLocation]] = {
    "spotify": {
        "terms": DocumentLocation(CONTRIB, "Spotify", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "Spotify", "Privacy Policy.md"),
    },
    "netflix": {
        "terms": DocumentLocation(CONTRIB, "Netflix", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "Netflix", "Privacy Policy.md"),
    },
    "lyft": {
        "terms": DocumentLocation(CONTRIB, "Lyft", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "Lyft", "Privacy Policy.md"),
    },
    "discord": {
        "terms": DocumentLocation(CONTRIB, "Discord", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "Discord", "Privacy Policy.md"),
    },
    "reddit": {
        "terms": DocumentLocation(CONTRIB, "Reddit", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "Reddit", "Privacy Policy.md"),
    },
    "zoom": {
        "terms": DocumentLocation(CONTRIB, "Zoom", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "Zoom", "Privacy Policy.md"),
    },
    "coursera": {
        "terms": DocumentLocation(CONTRIB, "Coursera", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "Coursera", "Privacy Policy.md"),
    },
    "paypal": {
        "terms": DocumentLocation(CONTRIB, "PayPal", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "PayPal", "Privacy Policy.md"),
    },
    "strava": {
        "terms": DocumentLocation(TOSDR, "Strava", "Terms of Service.md"),
        "privacy": DocumentLocation(TOSDR, "Strava", "Privacy Policy.md"),
    },
    "pinterest": {
        "terms": DocumentLocation(CONTRIB, "Pinterest", "Terms of Service.md"),
        "privacy": DocumentLocation(CONTRIB, "Pinterest", "Privacy Policy.md"),
    },
}


@dataclass(frozen=True)
class Source:
    """A git repository of policy documents, laid out as <service>/<document title>.md."""

    name: str
    url: str
    tag_prefix: str | None = None  # Clone the newest tag with this prefix, not the default branch.


SOURCES = (
    Source(CONTRIB, "https://github.com/OpenTermsArchive/contrib-versions"),
    Source(
        TOSDR,
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


def find_document(root: Path, location: DocumentLocation) -> Path | None:
    """Return the document's file, ignoring stubs too small to be a real policy."""
    path = root / location.folder / location.filename
    if path.is_file() and path.stat().st_size >= MIN_DOCUMENT_BYTES:
        return path
    return None


def copy_documents(clones: dict[str, tuple[Path, str]], raw_dir: Path) -> dict[str, dict[str, str]]:
    """Copy each document from its named source, and record its origin."""
    manifest: dict[str, dict[str, str]] = {}
    for company, locations in DOCUMENTS.items():
        for doc_type, location in locations.items():
            if location.source not in clones:
                continue
            root, ref = clones[location.source]
            if path := find_document(root, location):
                target = raw_dir / f"{company}_{doc_type}.md"
                shutil.copyfile(path, target)
                manifest[target.name] = {
                    "source": location.source,
                    "ref": ref,
                    "path": str(path.relative_to(root)),
                }
    return manifest


def print_report(manifest: dict[str, dict[str, str]]) -> None:
    for company in DOCUMENTS:
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
