import os
import time
import traceback

class GoogleSheetLogger:
    def __init__(self, credentials_file: str, sheet_name: str):
        self.enabled = False
        self.worksheet = None
        self._init_client(credentials_file, sheet_name)

    def _init_client(self, credentials_file: str, sheet_name: str):
        if not os.path.exists(credentials_file):
            print(f"❌ Sheets configuration mismatch: Missing file {credentials_file}")
            return

        try:
            import gspread
            from google.oauth2.service_account import Credentials
            from gspread.exceptions import SpreadsheetNotFound, APIError
        except ImportError:
            print("❌ Dependencies missing for automated network sync sheet integration.")
            return

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        try:
            creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
            client = gspread.authorize(creds)
            sheet = None
            
            for attempt in range(3):
                try:
                    sheet = client.open(sheet_name)
                    break 
                except SpreadsheetNotFound:
                    sheet = client.create(sheet_name)
                    break
                except APIError as e:
                    if "503" in str(e) and attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    raise e

            if not sheet:
                return

            self.worksheet = sheet.sheet1
            first_row = self.worksheet.row_values(1)
            
            if not first_row or first_row[0] != "Timestamp (UTC)":
                header = [
                    "Timestamp (UTC)", "Asset", "Current Price", "Action Taken",
                    "Decision Conf %", "Tech Signal", "ML Probability", "Units Held",
                    "Total Equity", "Portfolio Drawdown", "Order Executed",
                    "Market Regime", "Strategy Threshold", "Trade Note"
                ]
                self.worksheet.insert_row(header, 1)

            self.enabled = True
            print(f"🚀 Google Sheets communication interface fully established for: {sheet_name}")

        except Exception as e:
            print(f"❌ Initialization engine failure: {e}")
            self.enabled = False

    def log_row(self, row) -> int | None:
        if not self.enabled or self.worksheet is None:
            return None
        try:
            self.worksheet.append_row(row, value_input_option="USER_ENTERED")
            time.sleep(0.4)  # Small cooldown window buffer to let Google parse allocation length
            return len(self.worksheet.get_all_values())
        except Exception as exc:
            print(f"❌ Sheets append transmission crashed: {exc}")
            return None

    def update_row(self, row_index: int, row: list):
        if not self.enabled or self.worksheet is None or row_index is None:
            return
        try:
            cell_range = f"A{row_index}:N{row_index}"
            self.worksheet.update(range_name=cell_range, values=[row], value_input_option="USER_ENTERED")
            print(f"✨ Row {row_index} successfully updated in the cloud grid matrix.")
        except Exception as exc:
            print(f"❌ Sheets matrix modification gateway failure: {exc}")
