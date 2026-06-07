import traceback
import json
import os

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("[SheetLogger] gspread not installed — logging disabled.")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

HEADERS = [
    "timestamp", "symbol", "price", "action",
    "confidence", "tech_signal", "ml_prob",
    "qty", "equity", "drawdown",
    "regime", "threshold", "note"
]


class GoogleSheetLogger:
    def __init__(self, credentials_file: str, sheet_name: str):
        self.sheet = None
        if not GSPREAD_AVAILABLE:
            return
        try:
            creds = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
            # Use gspread.Client directly — gspread.authorize() is deprecated
            client = gspread.Client(auth=creds)
            client.session = None  # forces fresh session
            spreadsheet = client.open(sheet_name)
            self.sheet = spreadsheet.sheet1

            # Write headers if sheet is empty
            if self.sheet.row_count == 0 or not self.sheet.get("A1"):
                self.sheet.append_row(HEADERS, value_input_option="USER_ENTERED")

            print(f"[SheetLogger] Connected to '{sheet_name}'")
        except Exception as e:
            print(f"[SheetLogger] Init FAILED: {e}")
            traceback.print_exc()
            self.sheet = None

    def log_row(self, row: list):
        if self.sheet is None:
            print(f"[SheetLogger] (disabled) row={row}")
            return
        try:
            self.sheet.append_row(
                [str(v) for v in row],
                value_input_option="USER_ENTERED"
            )
            print(f"[SheetLogger] Logged: {row[1]} {row[3]}")
        except Exception as e:
            print(f"[SheetLogger] append_row FAILED: {e}")
            traceback.print_exc()