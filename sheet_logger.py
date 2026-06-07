import os
import time


class GoogleSheetLogger:
    def __init__(self, credentials_file: str, sheet_name: str):
        self.enabled = False
        self.worksheet = None
        self._init(credentials_file, sheet_name)

    def _init(self, credentials_file, sheet_name):
        if not os.path.exists(credentials_file):
            return

        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        client = gspread.authorize(creds)

        sheet = client.open(sheet_name)
        self.worksheet = sheet.sheet1
        self.enabled = True

    def log_row(self, row):
        if not self.enabled:
            return None

        self.worksheet.append_row(row, value_input_option="USER_ENTERED")
        time.sleep(0.3)
        return True