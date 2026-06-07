import traceback
import gspread
from google.oauth2.service_account import Credentials

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
        try:
            creds = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
            gc = gspread.authorize(creds)
            spreadsheet = gc.open(sheet_name)
            self.sheet = spreadsheet.sheet1

            # Fetch the actual populated content array of row 1
            first_row = self.sheet.row_values(1)
            
            # If the list is completely empty or row 1 column 1 has no text content
            if not first_row or not str(first_row[0]).strip():
                # v5 positional signature for appending headers
                self.sheet.append_row(HEADERS, value_input_option="USER_ENTERED")
                print("[SheetLogger] Headers written.")
            else:
                print(f"[SheetLogger] Existing headers verified: '{first_row[0]}'")

            print(f"[SheetLogger] Connected to '{sheet_name}'")

        except Exception as e:
            print(f"[SheetLogger] Init FAILED: {e}")
            traceback.print_exc()

    def log_row(self, row: list):
        if self.sheet is None:
            print(f"[SheetLogger] DISABLED — row: {row}")
            return
        try:
            string_row = [str(v) for v in row]
            
            # v5 sequential positioning sequence
            self.sheet.append_row(
                string_row,
                value_input_option="USER_ENTERED",
                table_range="A1",
            )
            print(f"[SheetLogger] ✓ Logged {row[1]} {row[3]}")
        except Exception as e:
            print(f"[SheetLogger] append_row FAILED: {e}")
            traceback.print_exc()
