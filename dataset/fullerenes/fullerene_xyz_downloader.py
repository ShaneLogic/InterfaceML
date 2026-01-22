import os
import re
import csv
import sys
import time
import math
import hashlib
import urllib.parse
import threading
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Configuration
# =========================

BASE = "https://nanotube.msu.edu/fullerene/"
INDEX_URL = urllib.parse.urljoin(BASE, "fullerene-isomers.html")
OUT_ROOT = "fullerene_xyz"

TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (compatible; FullereneXYZDownloader/2.0; +https://nanotube.msu.edu/)"

# Concurrency and throttling
MAX_WORKERS = 8              # Increase carefully (be polite to the server)
REQUESTS_PER_SECOND = 3.0    # Global rate limit across all threads
SLEEP_BETWEEN_PAGES = 0.2    # Delay between fetching C-pages

# Retry policy
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 0.8     # seconds; exponential backoff

# Output artifacts
URL_LIST_NAME = "_xyz_urls_by_C.txt"
FAILED_URLS_NAME = "_failed_urls.txt"
INDEX_CSV_NAME = "index.csv"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

# =========================
# Rate limiter (global)
# =========================

class RateLimiter:
    """
    Simple global token-bucket-like rate limiter using a shared next-allowed timestamp.
    Ensures average REQUESTS_PER_SECOND across all threads.
    """
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 1e-9)
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                sleep_for = self._next_allowed - now
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval

rate_limiter = RateLimiter(REQUESTS_PER_SECOND)

# =========================
# Data structures
# =========================

@dataclass
class DownloadJob:
    c_value: int
    url: str
    filename: str
    out_path: str

@dataclass
class FileRecord:
    c_value: int
    filename: str
    url: str
    atom_count: Optional[int]
    valid: bool
    reason: str
    size_bytes: int
    sha256: str

@dataclass
class DownloadSummary:
    ok: int
    skipped: int
    failed: int

# =========================
# HTTP helpers
# =========================

def http_get(url: str) -> requests.Response:
    """
    GET with global rate limiting + retries.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limiter.wait()
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            # Add small jitter without random (deterministic): sin-based
            jitter = 0.05 * abs(math.sin(attempt * 3.14159 / 7))
            time.sleep(backoff + jitter)
    raise RuntimeError(f"GET failed after retries: {url} :: {repr(last_err)}")

def fetch_html(url: str) -> str:
    return http_get(url).text

# =========================
# Parsing helpers
# =========================

def extract_c_pages(index_html: str) -> List[Tuple[int, str]]:
    """
    Extract all C-values and their corresponding fullerene.php?C=xx URLs
    from the index page.
    Returns list of (C_value, url).
    """
    soup = BeautifulSoup(index_html, "html.parser")
    pairs: Set[Tuple[int, str]] = set()

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        m = re.search(r'fullerene\.php\?C=(\d+)', href)
        if m:
            c_val = int(m.group(1))
            url = urllib.parse.urljoin(BASE, href)
            pairs.add((c_val, url))

    # Fallback: regex scan if structure changes
    if not pairs:
        for m in re.findall(r'fullerene\.php\?C=(\d+)', index_html):
            c_val = int(m)
            url = urllib.parse.urljoin(BASE, f"fullerene.php?C={c_val}")
            pairs.add((c_val, url))

    return sorted(pairs, key=lambda x: x[0])

def extract_xyz_urls_from_c_page(c_page_html: str, c_page_url: str) -> Set[str]:
    """
    Extract all .xyz URLs from a fullerene.php?C=xx page.
    Typically links look like: ./C84/C84-xxx.xyz
    """
    soup = BeautifulSoup(c_page_html, "html.parser")
    xyz: Set[str] = set()

    # Normal case: direct <a href="...xyz">
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if href.lower().endswith(".xyz"):
            xyz.add(urllib.parse.urljoin(c_page_url, href))

    # Fallback: raw text references to ./Cxx/...xyz
    for m in re.findall(r'(\./C\d+\/[^"\'\s<>]+\.xyz)', c_page_html, flags=re.IGNORECASE):
        xyz.add(urllib.parse.urljoin(c_page_url, m))

    # Last resort: infer xyz from png basenames if only png references exist
    if not xyz:
        for p in re.findall(r'(\./C\d+\/[^"\'\s<>]+\.png)', c_page_html, flags=re.IGNORECASE):
            xyz.add(urllib.parse.urljoin(c_page_url, p[:-4] + ".xyz"))

    return xyz

def filename_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = os.path.basename(path)
    return name if name else "unknown.xyz"

# =========================
# XYZ validation helpers
# =========================

_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def validate_xyz_bytes(data: bytes) -> Tuple[bool, Optional[int], str]:
    """
    Validate XYZ format:
      - Line 1: integer atom count N > 0
      - Line 2: comment (any)
      - Next N lines: element symbol + 3 floats (x y z)
      - Must have at least N+2 lines; extra trailing lines are treated as invalid by default here
        (you can relax this if needed).
    Returns (valid, atom_count, reason).
    """
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return (False, None, "DecodeError")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) < 3:
        return (False, None, "TooFewLines")

    # Atom count
    try:
        n = int(lines[0].split()[0])
    except Exception:
        return (False, None, "InvalidAtomCountLine")

    if n <= 0:
        return (False, n, "NonPositiveAtomCount")

    if len(lines) != n + 2:
        # Strict mode: require exact line count
        return (False, n, f"LineCountMismatch(expected={n+2}, got={len(lines)})")

    # Validate coordinates
    for i in range(2, 2 + n):
        parts = lines[i].split()
        if len(parts) < 4:
            return (False, n, f"BadAtomLineFields(line={i+1})")
        elem = parts[0]
        if not _ELEMENT_RE.match(elem):
            return (False, n, f"BadElementSymbol(line={i+1}, elem={elem})")
        x, y, z = parts[1], parts[2], parts[3]
        if not (_FLOAT_RE.match(x) and _FLOAT_RE.match(y) and _FLOAT_RE.match(z)):
            return (False, n, f"BadCoordinates(line={i+1})")

    return (True, n, "OK")

# =========================
# Download worker
# =========================

def download_job(job: DownloadJob) -> Tuple[Optional[FileRecord], Optional[str]]:
    """
    Download one XYZ file, save to disk, validate, and return FileRecord.
    Returns (record, failed_url_if_any).
    """
    # Skip if already exists and non-empty
    if os.path.exists(job.out_path) and os.path.getsize(job.out_path) > 0:
        # Read and validate existing file for index completeness
        try:
            with open(job.out_path, "rb") as f:
                data = f.read()
            valid, n, reason = validate_xyz_bytes(data)
            rec = FileRecord(
                c_value=job.c_value,
                filename=job.filename,
                url=job.url,
                atom_count=n,
                valid=valid,
                reason=("OK(existing)" if valid else f"{reason}(existing)"),
                size_bytes=len(data),
                sha256=sha256_bytes(data),
            )
            return rec, None
        except Exception as e:
            # Existing file but cannot read/validate -> treat as failed
            return None, job.url

    try:
        r = http_get(job.url)
        data = r.content

        # Save raw file
        os.makedirs(os.path.dirname(job.out_path), exist_ok=True)
        with open(job.out_path, "wb") as f:
            f.write(data)

        valid, n, reason = validate_xyz_bytes(data)
        rec = FileRecord(
            c_value=job.c_value,
            filename=job.filename,
            url=job.url,
            atom_count=n,
            valid=valid,
            reason=reason,
            size_bytes=len(data),
            sha256=sha256_bytes(data),
        )

        # If invalid, write a sidecar marker for quick inspection
        if not valid:
            with open(job.out_path + ".invalid.txt", "w", encoding="utf-8") as f:
                f.write(f"INVALID XYZ: {reason}\nURL: {job.url}\n")

        return rec, None

    except Exception as e:
        # Write an error marker for debugging
        os.makedirs(os.path.dirname(job.out_path), exist_ok=True)
        with open(job.out_path + ".error.txt", "w", encoding="utf-8") as f:
            f.write(f"FAILED: {job.url}\n{repr(e)}\n")
        return None, job.url

# =========================
# Index writer
# =========================

def write_index_csv(out_root: str, records: List[FileRecord]) -> str:
    """
    Write index.csv with columns:
      C, filename, url, atom_count, valid, reason, bytes, sha256
    """
    path = os.path.join(out_root, INDEX_CSV_NAME)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["C", "filename", "url", "atom_count", "valid", "reason", "bytes", "sha256"])
        for rec in sorted(records, key=lambda r: (r.c_value, r.filename)):
            w.writerow([
                rec.c_value,
                rec.filename,
                rec.url,
                "" if rec.atom_count is None else rec.atom_count,
                "1" if rec.valid else "0",
                rec.reason,
                rec.size_bytes,
                rec.sha256
            ])
    return path

def write_urls_by_c(out_root: str, urls_by_c: Dict[int, List[str]]) -> str:
    """
    Write a text file listing XYZ URLs grouped by C value.
    """
    path = os.path.join(out_root, URL_LIST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        for c in sorted(urls_by_c.keys()):
            f.write(f"[C{c}]\n")
            for u in sorted(set(urls_by_c[c])):
                f.write(u + "\n")
            f.write("\n")
    return path

# =========================
# Main
# =========================

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"Fetching index: {INDEX_URL}")
    index_html = fetch_html(INDEX_URL)

    c_pages = extract_c_pages(index_html)
    if not c_pages:
        print("ERROR: No C-pages found. The site structure may have changed.")
        sys.exit(1)

    print(f"Found {len(c_pages)} C-pages.")
    urls_by_c: Dict[int, List[str]] = {}
    all_jobs: List[DownloadJob] = []

    # 1) Crawl each C-page and collect XYZ URLs (strict mapping by page C)
    for idx, (c_val, c_url) in enumerate(c_pages, 1):
        print(f"[{idx}/{len(c_pages)}] Fetching C-page: C{c_val} -> {c_url}")
        html = fetch_html(c_url)
        xyz_urls = extract_xyz_urls_from_c_page(html, c_url)

        urls_by_c[c_val] = sorted(xyz_urls)
        print(f"  -> XYZ links found for C{c_val}: {len(xyz_urls)}")

        # Create download jobs under OUT_ROOT/Cxx/
        subdir = os.path.join(OUT_ROOT, f"C{c_val}")
        for u in xyz_urls:
            fn = filename_from_url(u)
            out_path = os.path.join(subdir, fn)
            all_jobs.append(DownloadJob(c_value=c_val, url=u, filename=fn, out_path=out_path))

        time.sleep(SLEEP_BETWEEN_PAGES)

    # Save URL listing for reproducibility
    url_list_path = write_urls_by_c(OUT_ROOT, urls_by_c)
    print(f"URL list written: {os.path.abspath(url_list_path)}")

    # De-duplicate jobs by (C, filename, url)
    uniq = {}
    for j in all_jobs:
        key = (j.c_value, j.filename, j.url)
        uniq[key] = j
    jobs = list(uniq.values())

    total_urls = sum(len(v) for v in urls_by_c.values())
    print(f"Total XYZ URLs (sum over C-pages): {total_urls}")
    print(f"Total download jobs (unique): {len(jobs)}")
    print(f"Concurrency: MAX_WORKERS={MAX_WORKERS}, RPS={REQUESTS_PER_SECOND}")

    # 2) Parallel download with global throttling
    records: List[FileRecord] = []
    failed_urls: List[str] = []
    ok = skipped = failed = 0

    # A file is "skipped" if it already exists and was validated; we infer it from record.reason suffix
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(download_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rec, failed_url = fut.result()
            if rec is not None:
                records.append(rec)
                if rec.reason.endswith("(existing)"):
                    skipped += 1
                else:
                    ok += 1
            if failed_url is not None:
                failed += 1
                failed_urls.append(failed_url)

            if i % 100 == 0 or i == len(futures):
                print(f"[progress] {i}/{len(futures)} ok={ok} skipped={skipped} failed={failed}")

    # 3) Write failed URLs (if any)
    if failed_urls:
        failed_path = os.path.join(OUT_ROOT, FAILED_URLS_NAME)
        with open(failed_path, "w", encoding="utf-8") as f:
            for u in sorted(set(failed_urls)):
                f.write(u + "\n")
        print(f"Failed URLs written: {os.path.abspath(failed_path)}")

    # 4) Write index.csv
    index_path = write_index_csv(OUT_ROOT, records)
    print(f"Index CSV written: {os.path.abspath(index_path)}")

    # 5) Final counts by directory (Cxx)
    per_c_counts: Dict[int, int] = {}
    per_c_valid: Dict[int, int] = {}
    for rec in records:
        per_c_counts[rec.c_value] = per_c_counts.get(rec.c_value, 0) + 1
        if rec.valid:
            per_c_valid[rec.c_value] = per_c_valid.get(rec.c_value, 0) + 1

    total_local_xyz = 0
    for c_val in sorted(urls_by_c.keys()):
        cdir = os.path.join(OUT_ROOT, f"C{c_val}")
        if not os.path.isdir(cdir):
            continue
        n_xyz = sum(1 for fn in os.listdir(cdir) if fn.lower().endswith(".xyz"))
        total_local_xyz += n_xyz

    print("----- SUMMARY -----")
    print(f"Output folder: {os.path.abspath(OUT_ROOT)}")
    print(f"Total XYZ URLs discovered (sum over C-pages): {total_urls}")
    print(f"Jobs executed (unique): {len(jobs)}")
    print(f"Downloaded OK (new): {ok}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Local .xyz files on disk (count): {total_local_xyz}")

    # Optional: show per-C breakdown for quick checks (only non-zero)
    print("Per-C breakdown (downloaded/valid):")
    for c_val in sorted(per_c_counts.keys()):
        print(f"  C{c_val}: {per_c_counts[c_val]} downloaded, {per_c_valid.get(c_val, 0)} valid")

    # Exit code: non-zero if failures occurred
    if failed > 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
