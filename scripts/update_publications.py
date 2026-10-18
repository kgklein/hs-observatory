#!/usr/bin/env python3
"""Update the HS Observatory MkDocs bibliography from a SciX/ADS library.

By default this script only READS the library. With --sync it first adds
records citing the two foundational HelioSwarm papers to the library.

Usage, from the repository root:
    python scripts/update_publications.py --dry-run
    python scripts/update_publications.py
    python scripts/update_publications.py --sync

Requires: python -m pip install requests
          export ADS_API_TOKEN=...  (never commit the token)
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import quote

import requests

API = "https://api.adsabs.harvard.edu/v1"
LIBRARY_ID = "QcoPEsHvSGG6T2NGKzFi7g"
FOUNDATIONAL = ("2019arXiv190305740K", "2023SSRv..219...74K")
FIELDS = "bibcode,title,author,year,pub,doi,identifier,property"
PAGE = Path(__file__).resolve().parents[1] / "docs" / "publications.md"
BLOCKS = ("FOUNDATIONAL", "CITATIONS")
PAGE_SIZE = 100
METADATA_BATCH = 20


class ADSClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def request(self, method: str, endpoint: str, **kwargs) -> dict:
        try:
            response = self.session.request(method, API + endpoint, timeout=45, **kwargs)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"ADS API request failed ({endpoint}): {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected ADS API response for {endpoint}")
        return data

    def search(self, query: str, fields: str = "bibcode") -> list[dict]:
        docs: list[dict] = []
        start = 0
        while True:
            data = self.request(
                "GET",
                "/search/query",
                params={
                    "q": query,
                    "fl": fields,
                    "rows": PAGE_SIZE,
                    "start": start,
                },
            )
            result = data.get("response", {})
            count = result.get("numFound")
            batch = result.get("docs")
            if not isinstance(count, int) or not isinstance(batch, list):
                raise RuntimeError("Search response lacked result count/documents")
            docs.extend(batch)
            start += len(batch)
            if start >= count:
                return docs
            if not batch:
                raise RuntimeError("ADS returned an incomplete search result")

    def library_bibcodes(self) -> list[str]:
        endpoint = f"/biblib/libraries/{LIBRARY_ID}"
        metadata = self.request("GET", endpoint)
        count = metadata.get("metadata", {}).get("num_documents")
        if not isinstance(count, int):
            raise RuntimeError("Library metadata does not contain num_documents")
        bibcodes: list[str] = []
        for start in range(0, count, PAGE_SIZE):
            data = self.request("GET", endpoint, params={"start": start, "rows": PAGE_SIZE})
            batch = data.get("documents")
            if not isinstance(batch, list):
                raise RuntimeError("Library response missing document list")
            bibcodes.extend(batch)
        if len(bibcodes) != count or any(not isinstance(x, str) for x in bibcodes):
            raise RuntimeError("Incomplete or invalid library document list")
        return list(dict.fromkeys(bibcodes))

    def sync_library(self) -> None:
        found: set[str] = set()
        for bibcode in FOUNDATIONAL:
            results = self.search(f"citations(bibcode:{bibcode})")
            citations = {doc["bibcode"] for doc in results if doc.get("bibcode")}
            if len(citations) != len(results):
                raise RuntimeError(f"Citation results missing bibcodes for {bibcode}")
            print(f"  {bibcode}: {len(citations)} citing records")
            found.update(citations)
        existing = set(self.library_bibcodes())
        new = sorted(found - existing)
        print(f"  {len(new)} new citing records to add")
        for index in range(0, len(new), PAGE_SIZE):
            batch = new[index : index + PAGE_SIZE]
            self.request(
                "POST", f"/biblib/documents/{LIBRARY_ID}", json={"bibcode": batch, "action": "add"}
            )

    def metadata(self, bibcodes: list[str]) -> dict[str, dict]:
        records: dict[str, dict] = {}
        for index in range(0, len(bibcodes), METADATA_BATCH):
            batch = bibcodes[index : index + METADATA_BATCH]
            # Short batches keep the ordinary search/query endpoint practical.
            query = "bibcode:(" + " OR ".join(batch) + ")"
            for doc in self.search(query, FIELDS):
                if doc.get("bibcode") in batch:
                    records[doc["bibcode"]] = doc
        return records


def plain(value: object) -> str:
    """Normalize display strings without allowing API-supplied Markdown/HTML."""
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]*>", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def md(value: object) -> str:
    return re.sub(r"([\\`*\[\]<>|_])", r"\\\1", plain(value))


def first(value: object) -> str:
    if isinstance(value, list):
        return plain(value[0]) if value else ""
    return plain(value)


def author_label(authors: object) -> str:
    if not isinstance(authors, list) or not authors:
        return "Unknown author"
    names = [plain(x).split(",", 1)[0].strip() for x in authors]
    if len(names) == 1:
        return md(names[0])
    if len(names) == 2:
        return f"{md(names[0])} & {md(names[1])}"
    return f"{md(names[0])} *et al.*"


def title_and_link(doc: dict) -> tuple[str, str]:
    title = first(doc.get("title")) or "Untitled publication"
    dois = doc.get("doi") or []
    doi = plain(dois[0]) if isinstance(dois, list) and dois else ""
    bibcode = str(doc["bibcode"])
    url = (
        ("https://doi.org/" + quote(doi, safe="/.-_"))
        if doi
        else ("https://ui.adsabs.harvard.edu/abs/" + quote(bibcode, safe="") + "/abstract")
    )
    return md(title), url


def entry(doc: dict) -> str:
    year = plain(doc.get("year"))
    title, url = title_and_link(doc)
    journal = md(doc.get("pub") or "")
    authors = author_label(doc.get("author"))
    suffix = f" *{journal}*." if journal else ""
    return f"- **{authors} ({year}).** [{title}]({url}).{suffix}"


def record_key(doc: dict) -> tuple:
    """Deduplicate records with matching DOI; otherwise retain unique bibcodes."""
    dois = doc.get("doi") or []
    if isinstance(dois, list) and dois and plain(dois[0]):
        return ("doi", plain(dois[0]).lower())
    return ("bibcode", doc["bibcode"])


def citations_section(docs: list[dict]) -> tuple[str, int]:
    # Prefer refereed records if multiple library entries share the same DOI.
    chosen: dict[tuple, dict] = {}
    for doc in sorted(
        docs, key=lambda d: ("REFEREED" not in (d.get("property") or []), str(d["bibcode"]))
    ):
        chosen.setdefault(record_key(doc), doc)
    years: dict[str, list[dict]] = {}
    for doc in chosen.values():
        years.setdefault(plain(doc["year"]), []).append(doc)
    sections: list[str] = []
    for year in sorted(years, key=int, reverse=True):
        sections.append(f"### {year}\n")
        items = sorted(
            years[year],
            key=lambda d: (
                plain((d.get("author") or [""])[0]).casefold(),
                first(d.get("title")).casefold(),
                d["bibcode"],
            ),
        )
        sections.extend(entry(doc) for doc in items)
        sections.append("")
    return "\n".join(sections).strip(), len(chosen)


def replace_block(contents: str, name: str, replacement: str) -> str:
    begin = f"<!-- BEGIN GENERATED {name} -->"
    end = f"<!-- END GENERATED {name} -->"
    if contents.count(begin) != 1 or contents.count(end) != 1:
        raise RuntimeError(f"Expected exactly one generated {name} block in {PAGE}")
    start = contents.index(begin) + len(begin)
    stop = contents.index(end)
    if stop < start:
        raise RuntimeError(f"Generated {name} markers are reversed")
    return contents[:start] + "\n" + replacement.strip() + "\n" + contents[stop:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync", action="store_true", help="Add newly citing papers to SciX library"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and summarize without writing the page"
    )
    args = parser.parse_args()
    token = os.environ.get("ADS_API_TOKEN")
    if not token:
        parser.error("Set ADS_API_TOKEN in your environment")
    if not PAGE.is_file():
        parser.error(f"Page not found: {PAGE}")
    source = PAGE.read_text(encoding="utf-8")
    for block in BLOCKS:
        replace_block(source, block, "test")  # validate markers before API activity

    client = ADSClient(token)
    if args.sync:
        if args.dry_run:
            parser.error("--sync and --dry-run cannot be used together")
        print("Finding new citing papers and syncing SciX library...")
        client.sync_library()
    bibcodes = client.library_bibcodes()
    metadata = client.metadata(list(dict.fromkeys([*FOUNDATIONAL, *bibcodes])))
    missing = sorted(set([*FOUNDATIONAL, *bibcodes]) - set(metadata))
    if missing:
        raise RuntimeError(
            f"ADS metadata missing for {len(missing)} bibcodes: {', '.join(missing[:10])}"
        )
    foundations = "\n".join(entry(metadata[code]) for code in FOUNDATIONAL)
    citing_docs = [metadata[code] for code in bibcodes if code not in FOUNDATIONAL]
    # Do not publish partial/invalid results.
    for doc in [*(metadata[code] for code in FOUNDATIONAL), *citing_docs]:
        if not str(doc.get("year", "")).isdigit() or not first(doc.get("title")):
            raise RuntimeError(f"Incomplete metadata for {doc.get('bibcode')}")
    citations, displayed = citations_section(citing_docs)
    if not citations:
        raise RuntimeError("Citing-paper bibliography would be empty; refusing to overwrite page")
    updated = replace_block(source, "FOUNDATIONAL", foundations)
    updated = replace_block(updated, "CITATIONS", citations)
    print(
        f"Library: {len(bibcodes)} records | citations: {len(citing_docs)} | "
        f"displayed: {displayed} | deduplicated: {len(citing_docs) - displayed}"
    )
    print(f"Page: {'changed' if updated != source else 'unchanged'} ({PAGE})")
    if args.dry_run:
        return 0
    if updated != source:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=PAGE.parent, prefix=".publications-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
        try:
            temporary.replace(PAGE)
        finally:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
