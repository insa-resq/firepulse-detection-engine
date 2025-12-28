import rasterio
from pathlib import Path

# Dossier contenant les TIFFs
INPUT_DIR = Path("../ts-satfire")

tif_files = list(INPUT_DIR.rglob("*.tif"))

print(f"Found {len(tif_files)} TIFF files\n")

for tif in tif_files:
    try:
        with rasterio.open(tif) as src:
            print(f"File: {tif.name}")
            print(f"  Number of bands: {src.count}")
            print(f"  Band indices: {list(range(1, src.count+1))}")
            print(f"  Band descriptions: {src.descriptions}\n")

    except Exception as e:
        print(f"Error reading {tif.name}: {e}")
