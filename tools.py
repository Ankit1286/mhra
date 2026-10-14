import time
import datetime
import xml.etree.ElementTree as ET

import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
MIN_INTERVAL = 0.34  # PubMed allows 3 requests/sec without an API key

# Reused later for the tool schema's enum, so the model can only pick valid [pt] values
ALLOWED_STUDY_TYPES = [
    "meta-analysis",
    "systematic review",
    "randomized controlled trial",
    "clinical trial",
    "observational study",
    "review",
]

MAX_RESULTS_LIMIT = 20  # Also the schema's maximum, so the model is told the same bound
MAX_ATTEMPTS = 3

session = requests.Session()
_last_request_time = 0.0
_cache = {}  # (endpoint, sorted params) -> successful response, for this process only


def _get(endpoint, params):
    """Rate-limited, cached GET with retries. Returns (response, None) on success or (None, error_message)."""
    global _last_request_time
    cache_key = (endpoint, tuple(sorted(params.items())))
    if cache_key in _cache:
        return _cache[cache_key], None

    error = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            time.sleep(attempt)  # back off 1s, then 2s
        wait = MIN_INTERVAL - (time.time() - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.time()

        try:
            resp = session.get(BASE_URL + endpoint, params=params, timeout=10)
        except requests.RequestException as e:
            error = f"Network error while contacting PubMed: {e}"
            continue

        # Rate limiting and server errors are temporary, so they are worth retrying
        if resp.status_code == 429:
            error = "PubMed rate limit exceeded. Try again shortly."
            continue
        if resp.status_code >= 500:
            error = f"PubMed returned HTTP {resp.status_code}. Try again shortly."
            continue
        # Other 4xx errors mean the request itself is wrong, so retrying won't help
        if resp.status_code >= 400:
            return None, f"PubMed rejected the request (HTTP {resp.status_code})."

        _cache[cache_key] = resp
        return resp, None
    return None, error


def search_pubmed(query, study_types=None, max_results=10, min_year=2015, max_year=None):
    if not query or not query.strip():
        return {"error": "Empty query. Provide search terms."}

    study_types = [t.lower() for t in (study_types or [])]
    invalid = [t for t in study_types if t not in ALLOWED_STUDY_TYPES]
    if invalid:
        return {"error": f"Invalid study types: {invalid}. Allowed: {ALLOWED_STUDY_TYPES}"}

    # The schema states this bound, but the model isn't guaranteed to respect it
    max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))

    term = query
    if study_types:
        pt_filter = " OR ".join(f"{t}[pt]" for t in study_types)
        term = f"{query} AND ({pt_filter})"

    search_params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
    }
    if min_year or max_year:
        search_params.update({
            "datetype": "pdat",
            "mindate": min_year or 2015,
            "maxdate": max_year or datetime.date.today().year,
        })

    resp, err = _get("esearch.fcgi", search_params)
    if err:
        return {"error": err}
    search_result = resp.json().get("esearchresult", {})
    if "ERROR" in search_result:
        return {"error": f"PubMed could not parse the query: {search_result['ERROR']}"}

    pmids = search_result.get("idlist", [])
    total_found = int(search_result.get("count", 0))
    if not pmids:
        return {
            "results": [],
            "note": "No matches. Try broader terms, fewer study-type filters, or a wider year range.",
            "total_found": 0,
        }

    resp, err = _get("esummary.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
    if err:
        return {"error": err}
    summaries = resp.json().get("result", {})

    results = []
    skipped = []
    for pmid in pmids:
        record = summaries.get(pmid)
        if not record or "error" in record:
            skipped.append(pmid)
            continue
        results.append({
            "pmid": pmid,
            "title": record.get("title", ""),
            "year": (record.get("sortpubdate") or "")[:4] or None,
            "pub_types": [p for p in record.get("pubtype", []) if p != "Journal Article"],
            "has_abstract": "Has Abstract" in record.get("attributes", []),
            "journal": record.get("fulljournalname", ""),
            "first_author": record.get("sortfirstauthor", ""),
        })

    output = {"total_found": total_found, "results": results}
    if skipped:
        output["note"] = f"Could not load summaries for PMIDs: {', '.join(skipped)}"
    return output


def fetch_abstracts(pmids, max_chars=2500):
    if not pmids:
        return {"error": "No PMIDs given. Pass a list of PMIDs from search results."}
    pmids = [str(p) for p in pmids]

    resp, err = _get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"})
    if err:
        return {"error": err}
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return {"error": "PubMed returned malformed XML."}

    articles = []
    for article in root.findall("PubmedArticle"):
        citation = article.find("MedlineCitation")
        art = citation.find("Article") if citation is not None else None
        if art is None:
            continue  # reported below as missing
        pmid = citation.findtext("PMID")

        # itertext() pulls in text nested inside <i>, <sup>, etc.
        title_el = art.find("ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Year lives in <Year> for most papers, but some only have <MedlineDate> like "2019 Nov-Dec"
        pub_date = art.find("Journal/JournalIssue/PubDate")
        year = None
        if pub_date is not None:
            year = pub_date.findtext("Year") or (pub_date.findtext("MedlineDate") or "")[:4] or None

        pub_types = [pt.text for pt in art.findall("PublicationTypeList/PublicationType")
                     if pt.text and pt.text != "Journal Article"]

        sections = []
        for section in art.findall("Abstract/AbstractText"):
            text = "".join(section.itertext()).strip()
            label = section.get("Label")
            sections.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(sections)

        if not abstract:
            abstract = "[No abstract available]"
        elif len(abstract) > max_chars:
            abstract = abstract[:max_chars].rsplit(" ", 1)[0] + " ...[truncated]"

        articles.append({
            "pmid": pmid,
            "title": title,
            "year": year,
            "pub_types": pub_types,
            "abstract": abstract,
        })

    output = {"articles": articles}
    found = {a["pmid"] for a in articles}
    missing = [p for p in pmids if p not in found]
    if missing:
        output["note"] = f"No record returned for PMIDs: {', '.join(missing)}"
    return output


if __name__ == "__main__":
    import json

    search = search_pubmed("EMDR PTSD", study_types=["meta-analysis"], max_results=5)
    print(json.dumps(search, indent=2, ensure_ascii=False))

    pmids = [r["pmid"] for r in search.get("results", [])[:2]]
    print(json.dumps(fetch_abstracts(pmids), indent=2, ensure_ascii=False))
