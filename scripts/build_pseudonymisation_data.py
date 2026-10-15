#!/usr/bin/env python3
"""Download and normalise vendored pseudonymisation datasets.

Run once on a machine with internet access (e.g. a Snellius login node):

    python scripts/build_pseudonymisation_data.py --data-dir /path/to/output

The four CSVs are written to <data-dir>/pseudonymisation/.  Pass that path as
`pseudonymisation_data_dir` in your TOML config to override the package's
placeholder data at runtime.
"""
import argparse
import csv
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile


def _get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} after error: {exc}")
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# 1. Dutch first names
# ---------------------------------------------------------------------------

def build_first_names(out_path: str) -> int:
    print("Fetching Dutch first names from Wikidata SPARQL …")
    sparql = """
SELECT DISTINCT ?name ?gender WHERE {
  ?item wdt:P31 wd:Q202444 ;
        rdfs:label ?name .
  FILTER(LANG(?name) = "nl")
  OPTIONAL {
    ?item wdt:P21 ?genderItem .
    VALUES (?genderItem ?g) { (wd:Q6581072 "f") (wd:Q6581097 "m") }
  }
  BIND(COALESCE(?g, "x") AS ?gender)
}
ORDER BY ?name
"""
    url = "https://query.wikidata.org/sparql"
    req = urllib.request.Request(
        f"{url}?query={urllib.parse.quote(sparql)}&format=json",
        headers={"User-Agent": "build_pseudonymisation_data/1.0 (research)"},
    )
    raw = urllib.request.urlopen(req, timeout=60).read()
    results = json.loads(raw)
    rows = []
    for binding in results["results"]["bindings"]:
        name = binding["name"]["value"].strip()
        gender = binding.get("gender", {}).get("value", "x").strip()[:1].lower()
        if name:
            rows.append({"name": name, "gender": gender or "x", "weight": 1})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "gender", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {len(rows)} first names")
    return len(rows)


# ---------------------------------------------------------------------------
# 2. Dutch surnames
# ---------------------------------------------------------------------------

def build_surnames(out_path: str) -> int:
    print("Fetching Dutch surnames from digitalheir/family-names-in-the-netherlands …")
    url = (
        "https://raw.githubusercontent.com/digitalheir/family-names-in-the-netherlands"
        "/master/top_1000_last_names_in_the_netherlands_2007.csv"
    )
    raw = _get(url).decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for row in reader:
        name = (row.get("natural name") or "").strip()
        try:
            weight = int(float(row.get("count in 2007") or 1))
        except (ValueError, TypeError):
            weight = 1
        if name:
            rows.append({"name": name, "weight": max(weight, 1)})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {len(rows)} surnames")
    return len(rows)


# ---------------------------------------------------------------------------
# 3. Dutch places (GeoNames)
# ---------------------------------------------------------------------------

def build_places(out_path: str) -> int:
    print("Fetching Dutch places from GeoNames NL.zip …")
    url = "https://download.geonames.org/export/dump/NL.zip"
    data = _get(url)
    rows = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("NL.txt") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 15:
                    continue
                feature_class = parts[6]  # column 7: feature class (P = populated place)
                if feature_class != "P":
                    continue
                name = parts[1].strip()  # column 2: name
                try:
                    population = int(parts[14]) if parts[14] else 0
                except (ValueError, IndexError):
                    population = 0
                if name:
                    rows.append({"name": name, "weight": max(population, 1)})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {len(rows)} places")
    return len(rows)


def write_geonames_notice(out_dir: str) -> None:
    notice = (
        "NOTICE\n"
        "======\n"
        "dutch_places.csv is derived from GeoNames data (https://www.geonames.org/),\n"
        "licensed under the Creative Commons Attribution 4.0 License:\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "Data downloaded from: https://download.geonames.org/export/dump/NL.zip\n"
    )
    with open(os.path.join(out_dir, "NOTICE"), "w", encoding="utf-8") as f:
        f.write(notice)


# ---------------------------------------------------------------------------
# 4. Dutch hospitals (Wikidata)
# ---------------------------------------------------------------------------

def build_hospitals(out_path: str) -> int:
    print("Fetching Dutch hospitals from Wikidata SPARQL …")
    sparql = """
SELECT DISTINCT ?name WHERE {
  ?h wdt:P31 wd:Q16917 ;
     wdt:P17 wd:Q55 ;
     rdfs:label ?name .
  FILTER(LANG(?name) = "nl")
}
ORDER BY ?name
"""
    url = "https://query.wikidata.org/sparql"
    req = urllib.request.Request(
        f"{url}?query={urllib.parse.quote(sparql)}&format=json",
        headers={"User-Agent": "build_pseudonymisation_data/1.0 (research)"},
    )
    raw = urllib.request.urlopen(req, timeout=60).read()
    results = json.loads(raw)
    rows = []
    for binding in results["results"]["bindings"]:
        name = binding["name"]["value"].strip()
        if name:
            rows.append({"name": name})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {len(rows)} hospitals")
    return len(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Output directory; CSVs are written to <data-dir>/pseudonymisation/.",
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "pseudonymisation")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing pseudonymisation data to: {out_dir}\n")

    build_first_names(os.path.join(out_dir, "dutch_first_names.csv"))
    build_surnames(os.path.join(out_dir, "dutch_surnames.csv"))
    build_places(os.path.join(out_dir, "dutch_places.csv"))
    write_geonames_notice(out_dir)
    build_hospitals(os.path.join(out_dir, "dutch_hospitals.csv"))

    print(f"\nDone. Data written to: {out_dir}")
    print("When --data-folder matches the --data-dir used above, no TOML config is needed.")
    print("(Set pseudonymisation_data_dir only if you store data elsewhere.)")


if __name__ == "__main__":
    main()
