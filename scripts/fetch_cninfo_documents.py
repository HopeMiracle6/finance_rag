from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cninfo_client import download_cninfo_pdf, iter_cninfo_announcements
from src.config import load_config, resolve_path
from src.utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--searchkey", default="")
    parser.add_argument("--stock", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--use-env-proxy", action="store_true")
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--raw-pdf-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    metadata_output = resolve_path(args.metadata_output or paths["cninfo_metadata_path"])
    raw_pdf_dir = resolve_path(args.raw_pdf_dir or paths["raw_pdf_dir"])

    records = []
    for metadata in iter_cninfo_announcements(
        start_date=args.start_date,
        end_date=args.end_date,
        page_size=args.page_size,
        max_pages=args.max_pages,
        searchkey=args.searchkey,
        stock=args.stock,
        category=args.category,
        sleep=args.sleep,
        use_env_proxy=args.use_env_proxy,
    ):
        records.append(
            download_cninfo_pdf(
                metadata,
                raw_pdf_dir=raw_pdf_dir,
                sleep=args.sleep,
                use_env_proxy=args.use_env_proxy,
            )
        )

    write_jsonl(metadata_output, records)
    print(f"cninfo metadata saved: {metadata_output}, records={len(records)}")
    print(f"raw pdf dir: {raw_pdf_dir}")


if __name__ == "__main__":
    main()
