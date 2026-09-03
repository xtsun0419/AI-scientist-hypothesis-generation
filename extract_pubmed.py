#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import tarfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests

IDCONV_API = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
OA_FCGI   = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
EFETCH    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# -------------------- 基础工具 --------------------

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def read_ids(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids

def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def normalize_pmcid(x: str) -> Optional[str]:
    x = (x or "").strip()
    m = re.match(r"^(PMC)?(\d+)$", x, re.IGNORECASE)
    if not m:
        return None
    return "PMC" + m.group(2)

def looks_like_pmid(x: str) -> bool:
    x = (x or "").strip()
    return bool(re.fullmatch(r"\d{1,9}", x))

def looks_like_doi(x: str) -> bool:
    x = (x or "").strip()
    return x.lower().startswith("10.") and ("/" in x)

def ftp_to_https(url: str) -> str:
    # oa.fcgi 常给 ftp://ftp.ncbi.nlm.nih.gov/...，转成 https 更好下
    if url.startswith("ftp://"):
        p = urlparse(url)
        if p.hostname == "ftp.ncbi.nlm.nih.gov":
            return "https://ftp.ncbi.nlm.nih.gov" + p.path
    return url

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------- 数据结构 --------------------

@dataclass
class PaperRec:
    requested_id: str
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""

    title: str = ""
    abstract: str = ""
    journal: str = ""
    year: str = ""

    has_fulltext_xml: bool = False
    fulltext_xml_path: str = ""
    license: str = ""
    note: str = ""


# -------------------- 网络函数 --------------------

def download_file(url: str, out_path: str, session: requests.Session, timeout: int = 180) -> None:
    safe_mkdir(os.path.dirname(out_path))
    tmp = out_path + ".part"
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, out_path)

def idconv_batch(ids: List[str], email: str, tool: str, session: requests.Session) -> Dict[str, dict]:
    """
    返回：alias_map[任意输入ID] = {pmid, pmcid, doi}
    也会把 pmid/pmcid/doi 作为别名键补进去，便于查找。
    """
    params = {"ids": ",".join(ids), "tool": tool, "email": email}
    r = session.get(IDCONV_API, params=params, timeout=60)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    alias_map: Dict[str, dict] = {}

    for rec in root.findall(".//record"):
        req = (rec.get("requested-id") or rec.get("orig-id") or "").strip()
        pmid = (rec.get("pmid") or "").strip()
        pmcid = (rec.get("pmcid") or "").strip()
        doi = (rec.get("doi") or "").strip()

        if pmcid:
            pmcid = normalize_pmcid(pmcid) or pmcid

        info = {"pmid": pmid, "pmcid": pmcid, "doi": doi}

        # requested-id
        if req:
            alias_map[req] = info

        # also alias by pmid/pmcid/doi (方便你输入混着来)
        if pmid:
            alias_map[pmid] = info
        if pmcid:
            alias_map[pmcid] = info
        if doi:
            alias_map[doi] = info

    return alias_map

def oa_links_for_pmcid(pmcid: str, session: requests.Session) -> dict:
    """
    返回：{"links": {fmt: href}, "license": "..."}
    """
    r = session.get(OA_FCGI, params={"id": pmcid}, timeout=60)
    # A 404 means the article is not available from the PMC OA subset. This is
    # an expected outcome, not a failed metadata extraction.
    if r.status_code == 404:
        return {"links": {}, "license": ""}
    r.raise_for_status()

    root = ET.fromstring(r.text)
    links = {}

    rec = root.find(".//record")
    lic = ""
    if rec is not None:
        lic = (rec.get("license") or "").strip()

    for link in root.findall(".//link"):
        fmt = (link.get("format") or "").lower()
        href = (link.get("href") or "").strip()
        if fmt and href:
            links[fmt] = href

    return {"links": links, "license": lic}

def extract_nxml_from_tgz(tgz_path: str) -> Optional[bytes]:
    with tarfile.open(tgz_path, "r:gz") as tf:
        nxml_members = [m for m in tf.getmembers() if m.name.lower().endswith(".nxml")]
        if not nxml_members:
            return None
        m = nxml_members[0]
        f = tf.extractfile(m)
        if not f:
            return None
        return f.read()

def fetch_pubmed_meta_batch(pmids: List[str], api_key: Optional[str], email: str, tool: str, session: requests.Session) -> Dict[str, dict]:
    if not pmids:
        return {}

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": tool,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    r = session.get(EFETCH, params=params, timeout=90)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    out = {}

    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        if not pmid:
            continue

        title = (art.findtext(".//ArticleTitle") or "").strip()

        abs_elems = art.findall(".//Abstract/AbstractText")
        if abs_elems:
            # 结构化摘要可能带 Label
            parts = []
            for x in abs_elems:
                txt = (x.text or "").strip()
                lab = (x.attrib.get("Label") or "").strip()
                if lab and txt:
                    parts.append(f"{lab}: {txt}")
                elif txt:
                    parts.append(txt)
            abstract = "\n".join(parts).strip()
        else:
            abstract = ""

        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = (art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or "").strip()

        doi = ""
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType") == "doi" and (aid.text or "").strip():
                doi = (aid.text or "").strip()
                break

        out[pmid] = {
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "doi": doi,
        }

    return out


# -------------------- 主流程 --------------------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="输入文件：每行一个 PMCID / PMID / DOI")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--email", required=True, help="联系邮箱（NCBI 建议填写）")
    ap.add_argument("--tool", default="al_alloy_rag", help="tool 名称")
    ap.add_argument("--api_key", default="", help="NCBI API key（可选）")
    ap.add_argument("--sleep", type=float, default=0.34, help="每次请求后 sleep 秒数")
    ap.add_argument("--idconv_chunk", type=int, default=200)
    ap.add_argument("--efetch_chunk", type=int, default=200)
    ap.add_argument("--keep_tgz", action="store_true", help="保留下载的 tgz（默认不保留）")
    args = ap.parse_args()

    ids_file = args.ids
    out_dir = args.out
    email = args.email
    tool = args.tool
    api_key = args.api_key.strip() or None
    sleep_s = float(args.sleep)

    ids = read_ids(ids_file)
    if not ids:
        print(f"[ERR] ids 文件为空：{ids_file}", file=sys.stderr)
        sys.exit(1)

    safe_mkdir(out_dir)
    xml_dir = os.path.join(out_dir, "fulltext_xml")
    dl_dir  = os.path.join(out_dir, "downloads")
    safe_mkdir(xml_dir)
    safe_mkdir(dl_dir)

    meta_path   = os.path.join(out_dir, "meta.jsonl")
    errors_path = os.path.join(out_dir, "errors.jsonl")

    # 清空旧输出，避免追加混乱
    for p in [meta_path, errors_path]:
        if os.path.exists(p):
            os.remove(p)

    session = requests.Session()
    session.headers.update({
        "User-Agent": f"pmc-xml-fetch/1.0 (contact: {email})",
        "From": email,
    })

    # 初始化记录
    recs: Dict[str, PaperRec] = {rid: PaperRec(requested_id=rid) for rid in ids}

    # 1) idconv：尽量补齐 pmid/pmcid/doi
    alias_map: Dict[str, dict] = {}
    print(f"[1/3] idconv mapping for {len(ids)} ids ...")
    for batch in chunked(ids, args.idconv_chunk):
        try:
            alias_map.update(idconv_batch(batch, email=email, tool=tool, session=session))
        except Exception as e:
            append_jsonl(errors_path, {"stage": "idconv", "ids": batch, "error": str(e)})
        time.sleep(sleep_s)

    # 回填 pmid/pmcid/doi
    for rid, rec in recs.items():
        # 如果是纯 PMCID/PMID/DOI，优先规范化
        n_pmcid = normalize_pmcid(rid)
        # Bare numeric identifiers are PubMed IDs by default. A PMCID must use
        # its PMC prefix; idconv still fills PMCID values for PMID/DOI input.
        if n_pmcid and rid.upper().startswith("PMC"):
            rec.pmcid = n_pmcid

        if looks_like_pmid(rid):
            rec.pmid = rid

        if looks_like_doi(rid):
            rec.doi = rid

        info = alias_map.get(rid) or alias_map.get(rec.pmcid) or alias_map.get(rec.pmid) or alias_map.get(rec.doi)
        if info:
            rec.pmid  = info.get("pmid", "")  or rec.pmid
            rec.pmcid = info.get("pmcid", "") or rec.pmcid
            rec.doi   = info.get("doi", "")   or rec.doi

        if rec.pmcid:
            rec.pmcid = normalize_pmcid(rec.pmcid) or rec.pmcid

    # 2) 全文 XML：有 PMCID 就尝试 oa.fcgi -> tgz -> nxml
    print("[2/3] try fetch fulltext XML from PMC OA tgz ...")
    fulltext_try, fulltext_ok = 0, 0

    for rid, rec in recs.items():
        if not rec.pmcid:
            rec.note = (rec.note + ";no_pmcid").strip(";")
            continue

        fulltext_try += 1
        pmcid = rec.pmcid

        try:
            oa = oa_links_for_pmcid(pmcid, session=session)
            rec.license = oa.get("license", "") or rec.license
            links = oa.get("links", {}) or {}

            # 只要 tgz（因为你要 XML）
            if "tgz" not in links:
                rec.note = (rec.note + ";no_tgz_or_not_in_oa_subset").strip(";")
                time.sleep(sleep_s)
                continue

            xml_path = os.path.join(xml_dir, f"{pmcid}.nxml")
            if os.path.exists(xml_path) and os.path.getsize(xml_path) > 0:
                rec.has_fulltext_xml = True
                rec.fulltext_xml_path = xml_path
                fulltext_ok += 1
                continue

            tgz_url = ftp_to_https(links["tgz"])
            tgz_path = os.path.join(dl_dir, f"{pmcid}.tar.gz")

            download_file(tgz_url, tgz_path, session=session, timeout=180)
            nxml_bytes = extract_nxml_from_tgz(tgz_path)

            if not nxml_bytes:
                rec.note = (rec.note + ";tgz_no_nxml").strip(";")
                append_jsonl(errors_path, {"stage": "extract_nxml", "pmcid": pmcid, "tgz": tgz_path, "error": "no .nxml found"})
                time.sleep(sleep_s)
                continue

            with open(xml_path, "wb") as f:
                f.write(nxml_bytes)

            rec.has_fulltext_xml = True
            rec.fulltext_xml_path = xml_path
            fulltext_ok += 1

            if not args.keep_tgz:
                try:
                    os.remove(tgz_path)
                except Exception:
                    pass

        except Exception as e:
            rec.note = (rec.note + ";fulltext_fail").strip(";")
            append_jsonl(errors_path, {"stage": "pmc_fulltext", "requested_id": rid, "pmcid": rec.pmcid, "error": str(e)})

        time.sleep(sleep_s)

    print(f"    fulltext tried: {fulltext_try}, success: {fulltext_ok}")

    # 3) 摘要兜底：对所有有 PMID 的记录 efetch 拉元数据/摘要
    print("[3/3] fetch PubMed title/abstract/meta via efetch ...")
    pmids = []
    seen = set()
    for rec in recs.values():
        if rec.pmid and rec.pmid not in seen:
            seen.add(rec.pmid)
            pmids.append(rec.pmid)

    meta_map: Dict[str, dict] = {}
    for batch in chunked(pmids, args.efetch_chunk):
        try:
            meta_map.update(fetch_pubmed_meta_batch(batch, api_key=api_key, email=email, tool=tool, session=session))
        except Exception as e:
            append_jsonl(errors_path, {"stage": "efetch_pubmed", "pmids": batch, "error": str(e)})
        time.sleep(sleep_s)

    # 写 meta.jsonl
    with open(meta_path, "w", encoding="utf-8") as f:
        for rid, rec in recs.items():
            if rec.pmid and rec.pmid in meta_map:
                m = meta_map[rec.pmid]
                rec.title   = m.get("title", "") or rec.title
                rec.abstract= m.get("abstract", "") or rec.abstract
                rec.journal = m.get("journal", "") or rec.journal
                rec.year    = m.get("year", "") or rec.year
                if not rec.doi:
                    rec.doi = m.get("doi", "") or rec.doi
            else:
                if not rec.pmid:
                    rec.note = (rec.note + ";no_pmid_for_abstract").strip(";")
                else:
                    rec.note = (rec.note + ";efetch_no_return").strip(";")

            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # stats
    total = len(recs)
    nxml = sum(1 for r in recs.values() if r.has_fulltext_xml)
    nabs = sum(1 for r in recs.values() if (r.abstract or "").strip())
    print("\n[Done]")
    print(f"  Total: {total}")
    print(f"  Fulltext XML: {nxml}")
    print(f"  Non-empty abstract: {nabs}")
    print(f"  Out dir: {out_dir}")
    print(f"    - {meta_path}")
    print(f"    - {xml_dir}/")
    print(f"    - {errors_path}")

if __name__ == "__main__":
    main()
