from pathlib import Path
import fitz
import docx
import pandas as pd
import re
from typing import Dict


# ==========================================================
# CLEANER
# ==========================================================

def clean_text(text: str) -> str:

    text = re.sub(r"\n+", "\n", text)

    text = re.sub(r"[ \t]+", " ", text)

    text = text.strip()

    return text


# ==========================================================
# PDF
# ==========================================================

def read_pdf(file_path: str) -> str:

    text = ""

    with fitz.open(file_path) as doc:

        for page in doc:

            text += page.get_text()

            text += "\n"

    return clean_text(text)


# ==========================================================
# DOCX
# ==========================================================

def read_word(file_path: str) -> str:

    doc = docx.Document(file_path)

    text = "\n".join(
        para.text
        for para in doc.paragraphs
    )

    return clean_text(text)


# ==========================================================
# TXT
# ==========================================================

def read_text_file(file_path: str) -> str:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    return clean_text(text)


# ==========================================================
# CSV
# ==========================================================

def read_csv(file_path: str) -> str:

    df = pd.read_csv(file_path)

    return clean_text(
        df.to_string(index=False)
    )


# ==========================================================
# MARKDOWN
# ==========================================================

def read_markdown(file_path: str) -> str:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    return clean_text(text)


# ==========================================================
# MAIN READER
# ==========================================================

def read_file(file_path: str) -> Dict:

    path = Path(file_path)

    if not path.exists():

        raise FileNotFoundError(
            f"{file_path} does not exist."
        )

    extension = path.suffix.lower()

    if extension == ".pdf":

        text = read_pdf(file_path)

    elif extension in [".doc", ".docx"]:

        text = read_word(file_path)

    elif extension == ".txt":

        text = read_text_file(file_path)

    elif extension == ".csv":

        text = read_csv(file_path)

    elif extension == ".md":

        text = read_markdown(file_path)

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return {
        "text": text,
        "metadata": {
            "source": path.name,
            "file_type": extension,
            "path": str(path),
            "file_size": path.stat().st_size
        }
    }
