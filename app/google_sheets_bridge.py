from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper process for Google Sheets API access.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-values")
    fetch_parser.add_argument("--service-account-file", required=True)
    fetch_parser.add_argument("--spreadsheet-id", required=True)
    fetch_parser.add_argument("--range", required=True)

    update_parser = subparsers.add_parser("batch-update")
    update_parser.add_argument("--service-account-file", required=True)
    update_parser.add_argument("--spreadsheet-id", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    service = build_sheets_service(Path(args.service_account_file).expanduser().resolve())

    if args.command == "fetch-values":
        payload = fetch_values(
            service=service,
            spreadsheet_id=args.spreadsheet_id,
            range_name=args.range,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    request_body = json.load(sys.stdin)
    batch_update_values(
        service=service,
        spreadsheet_id=args.spreadsheet_id,
        request_body=request_body,
    )
    return 0


def build_sheets_service(service_account_file: Path):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        str(service_account_file),
        scopes=[SHEETS_SCOPE],
    )
    return build("sheets", "v4", credentials=credentials, cache_discovery=False)


def fetch_values(service, spreadsheet_id: str, range_name: str) -> dict:
    return (
        service.spreadsheets()
        .values()
        .get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            majorDimension="ROWS",
        )
        .execute()
    )


def batch_update_values(service, spreadsheet_id: str, request_body: dict) -> None:
    (
        service.spreadsheets()
        .values()
        .batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=request_body,
        )
        .execute()
    )


def _configure_stdio() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


if __name__ == "__main__":
    raise SystemExit(main())
