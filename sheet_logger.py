import os
import time

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
            from gspread.exceptions import SpreadsheetNotFound, APIError
        except ImportError:
            print("Sheets disabled: install gspread + google-auth.")
            return

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        
        try:
            creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
            client = gspread.authorize(creds)

            # --- SMART RETRY LOGIC ---
            sheet = None
            for attempt in range(3):
                try:
                    sheet = client.open(sheet_name)
                    break 
                except SpreadsheetNotFound:
                    print(f"Sheet '{sheet_name}' not found. Attempting to create...")
                    # Only create if it actually doesn't exist
                    sheet = client.create(sheet_name)
                    break
                except APIError as e:
                    # If it's a 503 (Service Unavailable), wait and try again
                    if "503" in str(e) and attempt < 2:
                        print(f"Google Service busy (503). Retrying in {2**attempt}s...")
                        time.sleep(2**attempt)
                        continue
                    raise e # Re-raise if it's a different error (like your 403 quota)

            if not sheet:
                return

            self.worksheet = sheet.sheet1

            # Header Update
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
            print(f"Google Sheets logging enabled: {sheet_name}")

        except Exception as e:
            print(f"Failed to initialize Sheets: {e}")
            self.enabled = False

    def log_row(self, row):
        if not self.enabled or self.worksheet is None:
            return
        try:
            self.worksheet.append_row(row, value_input_option="USER_ENTERED")
        except Exception as exc:
            print(f"Sheets log failed: {exc}")
