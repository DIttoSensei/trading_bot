"""
Run this standalone to test ONLY the Google Sheet connection.
python test_sheet.py
"""
import os
import gspread
from datetime import datetime

CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
SHEET_NAME = os.getenv("GOOGLE_SHEETS_NAME", "Trading Log")

print(f"gspread version: {gspread.__version__}")
print(f"Credentials file: {CREDENTIALS_FILE}")
print(f"Sheet name: {SHEET_NAME}")
print()

# Step 1 — auth
try:
    gc = gspread.service_account(filename=CREDENTIALS_FILE)
    print("✓ Auth OK")
except Exception as e:
    print(f"✗ Auth FAILED: {e}")
    exit(1)

# Step 2 — open sheet
try:
    spreadsheet = gc.open(SHEET_NAME)
    print(f"✓ Opened spreadsheet: {spreadsheet.id}")
    print(f"  Title: {spreadsheet.title}")
    print(f"  URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")
except Exception as e:
    print(f"✗ Open sheet FAILED: {e}")
    exit(1)

# Step 3 — get sheet1
try:
    sheet = spreadsheet.sheet1
    print(f"✓ Sheet1 title: {sheet.title}")
    print(f"  Row count: {sheet.row_count}")
    print(f"  Col count: {sheet.col_count}")
except Exception as e:
    print(f"✗ Get sheet1 FAILED: {e}")
    exit(1)

# Step 4 — read A1
try:
    a1 = sheet.acell("A1").value
    print(f"✓ A1 value: '{a1}'")
except Exception as e:
    print(f"✗ Read A1 FAILED: {e}")
    exit(1)

# Step 5 — write a test row directly to A2 using update (not append)
try:
    test_row = [datetime.utcnow().isoformat(), "TEST", "direct_update", "works"]
    sheet.update("A2", [test_row])
    print(f"✓ Direct update to A2 OK — check your sheet NOW for row 2")
except Exception as e:
    print(f"✗ Direct update FAILED: {e}")

# Step 6 — append_row with table_range
try:
    test_row2 = [datetime.utcnow().isoformat(), "TEST", "append_row", "works"]
    result = sheet.append_row(
        test_row2,
        value_input_option="USER_ENTERED",
        table_range="A1",
    )
    print(f"✓ append_row OK — result: {result}")
except Exception as e:
    print(f"✗ append_row FAILED: {e}")

# Step 7 — read back what's there
try:
    all_values = sheet.get_all_values()
    print(f"\n✓ Total rows with data: {len(all_values)}")
    for i, row in enumerate(all_values[:5]):
        print(f"  Row {i+1}: {row}")
except Exception as e:
    print(f"✗ Read all values FAILED: {e}")