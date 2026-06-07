"""
sheet_logger.py
Append-only Google Sheets logger.
NO update_row. NO row indexing. Single append per trade.

Setup:
  pip install gspread google-auth
  credentials.json = service account key from Google Cloud Console
  Share the sheet with the service account email.
"""
import traceback

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("[SheetLogger] gspread not installed — sheet logging disabled.")


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class GoogleSheetLogger:
    def __init__(self, credentials_file: str, sheet_name: str):
        self.sheet = None
        if not GSPREAD_AVAILABLE:
            return
        try:
            creds = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
            client = gspread.authorize(creds)
            spreadsheet = client.open(sheet_name)
            self.sheet = spreadsheet.sheet1
            print(f"[SheetLogger] Connected to '{sheet_name}'")
        except Exception as e:
            print(f"[SheetLogger] Init failed (sheet logging disabled): {e}")
            self.sheet = None

    def log_row(self, row: list):
        """
        Append a single row. Silently skips if sheet unavailable.
        row = [timestamp, symbol, price, action, confidence,
               tech_signal, ml_prob, qty, equity, drawdown,
               regime, threshold, note]
        """
        if self.sheet is None:
            print(f"[SheetLogger] (no sheet) {row}")
            return
        try:
            self.sheet.append_row(row, value_input_option="USER_ENTERED")
        except Exception as e:
            print(f"[SheetLogger] append_row failed: {e}")
            traceback.print_exc()