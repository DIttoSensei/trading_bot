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

            # Check if cell A1 is empty
            first_cell = self.sheet.acell("A1").value
            if not first_cell or not str(first_cell).strip():
                # v5 signature: values must be passed as the first positional argument
                self.sheet.append_row(HEADERS, value_input_option="USER_ENTERED")
                print("[SheetLogger] Headers written.")
            else:
                print(f"[SheetLogger] Existing headers verified: '{first_cell}'")

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
            # v5 signature: string_row is passed positionally as the first argument
            self.sheet.append_row(
                string_row,
                value_input_option="USER_ENTERED",
                table_range="A1",
            )
            print(f"[SheetLogger] ✓ Logged {row[1]} {row[3]}")
        except Exception as e:
            print(f"[SheetLogger] append_row FAILED: {e}")
            traceback.print_exc()
