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
        
        # Readable Header Update
        # We check the first cell. If it's empty or not our expected header, we write it.
        try:
            first_row = self.worksheet.row_values(1)
            if not first_row or first_row[0] != "Timestamp (UTC)":
                header = [
                    "Timestamp (UTC)",
                    "Asset",            # Changed from 'symbol' for readability
                    "Current Price",
                    "Action Taken",     # BUY/SELL/HOLD
                    "Decision Conf %",  # Easier to read than 'confidence'
                    "Tech Signal",
                    "ML Probability",
                    "Units Held",
                    "Total Equity",
                    "Portfolio Drawdown",
                    "Order Executed",   # Boolean 1/0
                    "Market Regime",
                    "Strategy Threshold",
                    "Trade Note"
                ]
                # If the sheet is brand new, append_row works. 
                # If we're updating an old sheet, we use update('A1', [header])
                self.worksheet.insert_row(header, 1)
        except Exception as e:
            print(f"Header initialization error: {e}")

        self.enabled = True
        print(f"Google Sheets logging enabled: {sheet_name}")

    def log_row(self, row):
        if not self.enabled or self.worksheet is None:
            return
        try:
            # We use USER_ENTERED so numbers look like numbers and dates like dates
            self.worksheet.append_row(row, value_input_option="USER_ENTERED")
        except Exception as exc:
            print(f"Sheets log failed: {exc}")