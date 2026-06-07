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
            print(f"❌ Sheets disabled: missing credentials file at path: {credentials_file}")
            return

        try:
            import gspread
            from google.oauth2.service_account import Credentials
            from gspread.exceptions import SpreadsheetNotFound, APIError
        except ImportError:
            print("❌ Sheets running in passive state: missing gspread + google-auth dependencies.")
            return

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        try:
            print(f"📡 Initializing Google credentials from {credentials_file}...")
            creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
            client = gspread.authorize(creds)

            sheet = None
            for attempt in range(3):
                try:
                    print(f"🔍 Attempting to open sheet: '{sheet_name}' (Attempt {attempt + 1}/3)...")
                    sheet = client.open(sheet_name)
                    print(f"✅ Found and opened existing spreadsheet: '{sheet_name}'")
                    break 
                except SpreadsheetNotFound:
                    print(f"⚠️ Sheet '{sheet_name}' not found on this account. Attempting to create a new one...")
                    try:
                        sheet = client.create(sheet_name)
                        print(f"⚙️ Created brand-new sheet named '{sheet_name}' in service account drive.")
                    except Exception as create_err:
                        print(f"❌ Failed to create sheet automatically: {create_err}")
                    break
                except APIError as e:
                    if "503" in str(e) and attempt < 2:
                        print(f"⏳ Google Service busy (503). Retrying in {2**attempt}s...")
                        time.sleep(2**attempt)
                        continue
                    print(f"❌ Google API Error encountered during target open: {e}")
                    raise e

            if not sheet:
                print("❌ Initialization aborted: Spreadsheet object could not be resolved.")
                return

            print("📂 Accessing default worksheet tab (sheet1)...")
            self.worksheet = sheet.sheet1

            print("📝 Checking for structural header row matrix...")
            first_row = self.worksheet.row_values(1)
            if not first_row or first_row[0] != "Timestamp (UTC)":
                header = [
                    "Timestamp (UTC)", "Asset", "Current Price", "Action Taken",
                    "Decision Conf %", "Tech Signal", "ML Probability", "Units Held",
                    "Total Equity", "Portfolio Drawdown", "Order Executed",
                    "Market Regime", "Strategy Threshold", "Trade Note"
                ]
                print("📥 Header row missing or invalid. Injecting standardized column keys...")
                self.worksheet.insert_row(header, 1)
            else:
                print("✅ Standardized header row structure validated.")

            self.enabled = True
            print(f"🚀 SUCCESS: Google Sheets logger fully initialized and enabled for target: '{sheet_name}'")

        except Exception as e:
            print(f"❌ Critical Failure inside Google Sheets initialization script: {e}")
            traceback.print_exc()
            self.enabled = False

    def log_row(self, row):
        # 🔍 CRITICAL TRACKING GATEWAY
        if not self.enabled or self.worksheet is None:
            print(f"⚠️ [LOGGER DROPOUT] log_row bypassed! State flags: enabled={self.enabled}, worksheet_loaded={self.worksheet is not None}")
            return
            
        try:
            print(f"📤 Sending row payload to Google Sheets API grid...")
            self.worksheet.append_row(row, value_input_option="USER_ENTERED")
            print("✨ Google Sheets API successfully committed and rendered row payload.")
        except Exception as exc:
            print(f"❌ Google Sheets API transmission crashed mid-flight: {exc}")
            traceback.print_exc()
