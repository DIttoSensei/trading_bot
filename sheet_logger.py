import os
import json
from google.oauth2.service_account import Credentials
import googleapiclient.discovery

class GoogleSheetLogger:
    """
    Flushes tabular execution lines directly to remote Google Spreadsheet frames.
    """
    def __init__(self, credentials_file: str, sheet_name: str):
        self.enabled = False
        if not os.path.exists(credentials_file):
            print(f"⚠️ Google Credentials not located at {credentials_file}. Bypassing cloud logs.")
            return
            
        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
            self.service = googleapiclient.discovery.build("sheets", "v4", credentials=creds)
            
            # Extract spreadsheet IDs from named parameters
            self.spreadsheet_name = sheet_name
            # Fallback mock setup for safe runtime bypass if spreadsheet ID is not fully populated
            self.spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID", "")
            if self.spreadsheet_id:
                self.enabled = True
        except Exception as e:
            print(f"⚠️ Failed initialization of Google Sheets Framework: {e}")

    def log_row(self, row_data: list):
        if not self.enabled:
            return
        try:
            # Sanitize matrix into primitive string fields to avoid payload translation rejections
            clean_row = [str(x) for x in row_data]
            body = {"values": [clean_row]}
            
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range="Sheet1!A:A",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body=body
            ).execute()
        except Exception as e:
            print(f"❌ Cloud execution sheet sync failure: {e}")