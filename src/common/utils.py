from pathlib import Path
from typing import TypedDict

import rasterio
from rasterio import CRS
from rasterio.warp import transform

GeoTiffMetadata = TypedDict(
    "GeoTiffMetadata",
    {
        "width": int,
        "height": int,
        "latitude": float,
        "longitude": float
    }
)

def get_geotiff_metadata(image_path: str | Path) -> GeoTiffMetadata:
    """
    Extracts metadata (size, coordinates) from a GeoTIFF.
    """
    with rasterio.open(image_path) as src:
        # Get bounds in the image's native CRS (Coordinate Reference System)
        bounds = src.bounds

        # Calculate center point in native CRS
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2

        # Transform to WGS84 (Latitude/Longitude) if not already
        # EPSG:4326 is the standard for Lat/Lon
        if src.crs != "EPSG:4326":
            longitude, latitude = transform(
                src_crs=src.crs,
                dst_crs=CRS.from_epsg("4326"),
                xs=[center_x],
                ys=[center_y]
            )
            return {
                "width": src.width,
                "height": src.height,
                "latitude": latitude[0],
                "longitude": longitude[0]
            }
        else:
            return {
                "width": src.width,
                "height": src.height,
                "latitude": center_y,
                "longitude": center_x
            }
