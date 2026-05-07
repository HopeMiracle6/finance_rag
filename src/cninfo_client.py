from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests

from src.schema import CninfoDocumentMetadata
from src.utils import ensure_dir, stable_id


CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC_BASE = "http://static.cninfo.com.cn"


def safe_file_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:120]


def normalize_publish_date(value: object) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp).date().isoformat()
    text = str(value).strip()
    if text.isdigit():
        return normalize_publish_date(int(text))
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return text[:10] if text else None


def infer_report_type(title: str | None, fallback: str | None = None) -> str | None:
    if fallback:
        return fallback
    title = title or ""
    rules = [
        ("投资者关系活动记录表", "投资者关系活动记录表"),
        ("年度报告", "年度报告"),
        ("半年度报告", "半年度报告"),
        ("第一季度报告", "季度报告"),
        ("第三季度报告", "季度报告"),
        ("季度报告", "季度报告"),
        ("风险提示", "风险提示公告"),
        ("临时公告", "临时公告"),
    ]
    for keyword, report_type in rules:
        if keyword in title:
            return report_type
    return "临时公告"


def normalize_cninfo_item(item: dict, pdf_path: str | Path | None = None) -> CninfoDocumentMetadata:
    adjunct_url = item.get("adjunctUrl") or item.get("adjunct_url") or ""
    source_url = item.get("pdf_url") or item.get("source_url") or ""
    if not source_url and adjunct_url:
        source_url = f"{CNINFO_STATIC_BASE}/{adjunct_url.lstrip('/')}"

    title = (item.get("announcementTitle") or item.get("title") or "").replace("<em>", "").replace("</em>", "")
    stock_code = item.get("secCode") or item.get("sec_code") or item.get("stock_code")
    company_name = item.get("secName") or item.get("sec_name") or item.get("company_name")
    publish_date = normalize_publish_date(item.get("announcementTime") or item.get("announcement_time") or item.get("publish_date"))
    report_type = infer_report_type(title, item.get("event_type") or item.get("report_type"))
    raw_id = item.get("id") or stable_id(stock_code, title, publish_date, source_url)
    file_name = f"{safe_file_stem(str(raw_id))}.pdf"
    if pdf_path:
        file_name = Path(pdf_path).name

    return CninfoDocumentMetadata(
        doc_id=str(raw_id),
        file_name=file_name,
        company_name=company_name,
        stock_code=stock_code,
        report_type=report_type,
        publish_date=publish_date,
        source_url=source_url or None,
        pdf_path=str(pdf_path) if pdf_path else item.get("pdf_path"),
        title=title or None,
        source="cninfo",
        metadata={k: v for k, v in item.items() if k not in {"text"}},
    )


def iter_cninfo_announcements(
    start_date: str,
    end_date: str,
    page_size: int = 30,
    max_pages: int = 10,
    searchkey: str = "",
    stock: str = "",
    category: str = "",
    sleep: float = 0.5,
    use_env_proxy: bool = False,
) -> Iterable[CninfoDocumentMetadata]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
        "X-Requested-With": "XMLHttpRequest",
    }
    session = requests.Session()
    session.trust_env = use_env_proxy
    for page_num in range(1, max_pages + 1):
        data = {
            "pageNum": page_num,
            "pageSize": page_size,
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": stock,
            "searchkey": searchkey,
            "secid": "",
            "category": category,
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        response = session.post(CNINFO_QUERY_URL, data=data, headers=headers, timeout=20)
        response.raise_for_status()
        announcements = response.json().get("announcements") or []
        if not announcements:
            break
        for item in announcements:
            yield normalize_cninfo_item(item)
        time.sleep(sleep)


def download_cninfo_pdf(
    metadata: CninfoDocumentMetadata,
    raw_pdf_dir: str | Path,
    sleep: float = 0.2,
    use_env_proxy: bool = False,
) -> CninfoDocumentMetadata:
    if not metadata.source_url:
        return metadata
    output_dir = ensure_dir(raw_pdf_dir)
    output_path = output_dir / metadata.file_name
    if not output_path.exists():
        session = requests.Session()
        session.trust_env = use_env_proxy
        response = session.get(metadata.source_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        time.sleep(sleep)
    data = metadata.model_dump()
    data["pdf_path"] = str(output_path)
    data["file_name"] = output_path.name
    return CninfoDocumentMetadata(**data)


def copy_pdf_to_raw_dir(
    metadata: CninfoDocumentMetadata,
    source_pdf_path: str | Path,
    raw_pdf_dir: str | Path,
) -> CninfoDocumentMetadata:
    source = Path(source_pdf_path)
    output_dir = ensure_dir(raw_pdf_dir)
    target = output_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    data = metadata.model_dump()
    data["pdf_path"] = str(target)
    data["file_name"] = target.name
    return CninfoDocumentMetadata(**data)
