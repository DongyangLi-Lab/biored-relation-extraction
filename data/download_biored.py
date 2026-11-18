"""Fetch the three original BioRED BioC JSON splits from NCBI."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ARCHIVE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"
EXPECTED_SHA256 = "c3032230bd89d22a0923d0df6ae943bc8ea37fba7e42dafa7a8dec21bac02d47"
OUTPUT_DIR = Path(__file__).resolve().parent / "raw" / "BioRED"


def main():
    with urlopen(ARCHIVE_URL, timeout=60) as response:
        archive = response.read()
    if sha256(archive).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("The NCBI BioRED archive does not match the expected SHA-256.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(archive)) as source:
        for split in ("Train", "Dev", "Test"):
            name = f"{split}.BioC.JSON"
            target = OUTPUT_DIR / name
            target.write_bytes(source.read(f"BioRED/{name}"))
            print(f"Saved {target}")


if __name__ == "__main__":
    main()
