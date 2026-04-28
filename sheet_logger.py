import os


class GoogleSheetLogger:
    def __init__(self, credentials_file: str, sheet_name: str):
        self.enabled = False
        self.worksheet = None
        self._init_client(credentials_file, sheet_name)

    def _init_client(self, credentials_file: str, sheet_name: str):
        if not os.path.exists(credentials_file):
            print(f"Sheets disabled: missing credentials file: {credentials_file}")
            return
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except Exception:
            print("Sheets disabled: install gspread + google-auth.")
            return

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        client = gspread.authorize(creds)

        try:
            sheet = client.open(sheet_name)
        except Exception:
            sheet = client.create(sheet_name)

        self.worksheet = sheet.sheet1
        if self.worksheet.row_count < 2 or not self.worksheet.row_values(1):
            self.worksheet.append_row(
                [
                    "timestamp_utc",
                    "symbol",
                    "price",
                    "action",
                    "confidence",
                    "tech_signal",
                    "ml_prob",
                    "position_qty",
                    "equity",
                    "drawdown",
                    "traded",
                    "regime",
                    "threshold",
                    "note",
                ]
            )
        self.enabled = True
        print(f"Google Sheets logging enabled: {sheet_name}")

    def log_row(self, row):
        if not self.enabled or self.worksheet is None:
            return
        try:
            self.worksheet.append_row(row, value_input_option="RAW")
        except Exception as exc:
            print(f"Sheets log failed: {exc}")
