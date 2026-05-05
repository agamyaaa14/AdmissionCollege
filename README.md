# BCWCC Admission Helper

Simple Streamlit app for admission staff to clean up student photos, signatures, and scanned marks card PDFs before uploading to UUCMS.

## Features

- Automatic photo validation and resizing for passport-style uploads
- Photo and signature clipboard paste support
- Automatic signature validation, trimming, and compression
- Automatic PDF compression flow for marks cards
- Download buttons for processed outputs
- Download filenames use the student first name when provided

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The photo and signature checks use lightweight local heuristics.
- PDF compression is raster-based, so lower DPI gives smaller files but may reduce clarity.
- The app uses fixed defaults so staff do not need to adjust settings for each file.
- For paste support, click the paste area and press Ctrl+V or Cmd+V with an image already in the clipboard.
- Enter only the student's first name; downloads will be named like `name_photo`, `name_signature`, and `name_marks_card`.
