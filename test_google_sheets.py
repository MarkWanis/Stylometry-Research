import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_file(
    "credentials.json",
    scopes=SCOPES
)

client = gspread.authorize(credentials)

spreadsheet = client.open("Training Model Results")

worksheet = spreadsheet.sheet1

worksheet.update("A25", [["Hello from Python!"]])

print("Successfully connected to Google Sheets!")