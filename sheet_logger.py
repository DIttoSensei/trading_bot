import traceback
import gspread

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
            # service_account() works in gspread 5.x AND 6.x
            gc = gspread.service_account(filename=credentials_file)
            spreadsheet = gc.open(sheet_name)
            self.sheet = spreadsheet.sheet1

            # Write headers if A1 is empty
            first = self.sheet.acell("A1").value
            if not first:
                self.sheet.append_row(HEADERS, value_input_option="USER_ENTERED")
                print("[SheetLogger] Headers written.")

            print(f"[SheetLogger] Connected to '{sheet_name}' — id: {spreadsheet.id}")

        except Exception as e:
            print(f"[SheetLogger] Init FAILED: {e}")
            traceback.print_exc()

    def log_row(self, row: list):
        if self.sheet is None:
            print(f"[SheetLogger] DISABLED — row not saved: {row}")
            return
        try:
            result = self.sheet.append_row(
                [str(v) for v in row],
                value_input_option="USER_ENTERED"
            )
            updated = result.get("updates", {}).get("updatedRows", "?")
            print(f"[SheetLogger] ✓ Logged {row[1]} {row[3]} — updatedRows={updated}")
        except Exception as e:
            print(f"[SheetLogger] append_row FAILED: {e}")
            traceback.print_exc()