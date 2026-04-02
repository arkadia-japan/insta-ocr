from app.google_sheets_bridge import build_parser, list_sheets


class _FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeSpreadsheets:
    def __init__(self, payload):
        self._payload = payload
        self.last_spreadsheet_id = None
        self.last_fields = None

    def get(self, *, spreadsheetId, fields):
        self.last_spreadsheet_id = spreadsheetId
        self.last_fields = fields
        return _FakeRequest(self._payload)


class _FakeService:
    def __init__(self, payload):
        self._spreadsheets = _FakeSpreadsheets(payload)

    def spreadsheets(self):
        return self._spreadsheets


def test_build_parser_accepts_list_sheets_command() -> None:
    args = build_parser().parse_args(
        [
            "list-sheets",
            "--service-account-file",
            "service-account.json",
            "--spreadsheet-id",
            "spreadsheet-id",
        ]
    )

    assert args.command == "list-sheets"
    assert args.spreadsheet_id == "spreadsheet-id"


def test_list_sheets_returns_sheet_titles_and_metadata() -> None:
    service = _FakeService(
        {
            "properties": {"title": "運用ブック"},
            "sheets": [
                {"properties": {"sheetId": 10, "title": "投稿データA", "index": 0}},
                {"properties": {"sheetId": 11, "title": "投稿データB", "index": 1}},
            ],
        }
    )

    payload = list_sheets(service=service, spreadsheet_id="spreadsheet-id")

    assert service.spreadsheets().last_spreadsheet_id == "spreadsheet-id"
    assert payload == {
        "spreadsheet_title": "運用ブック",
        "sheets": [
            {"sheet_id": 10, "title": "投稿データA", "index": 0},
            {"sheet_id": 11, "title": "投稿データB", "index": 1},
        ],
    }
