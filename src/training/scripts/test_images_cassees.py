import rasterio
import numpy as np
from pathlib import Path

# Dossier contenant les TIFFs
INPUT_DIR = Path("../ts-satfire")

tif_files = list(INPUT_DIR.rglob("*.tif"))

print(f"Found {len(tif_files)} TIFF files\n")

for tif in tif_files:
    try:
        with rasterio.open(tif) as src:

            if tif.name == "2017-04-29_VIIRS_Day.tif" or tif.name == "2017-04-28_VIIRS_Day.tif" or tif.name == "2017-04-30_VIIRS_Day.tif" or tif.name == "2017-05-14_VIIRS_Day.tif": 
                print(f"File: {tif.name}")
                print(f"  Number of bands: {src.count}")
                print(f"  Band indices: {list(range(1, src.count+1))}")
                print(f"  Band descriptions: {src.descriptions}\n")
                m11 = src.read(6)
                af = src.read(7)

                # Vérifie si la bande est vide (tout 0 ou tout NaN)
                m11_empty = np.all(m11 == 0) or np.all(np.isnan(m11))
                af_empty = np.all(af == 0) or np.all(np.isnan(af))

                # Affiche le résultat
                print(f"{tif.name}: m11 empty = {m11_empty}, af empty = {af_empty}, shape = {m11.shape}")

    except Exception as e:
        print(f"Error reading {tif.name}: {e}")
