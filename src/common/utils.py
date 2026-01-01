from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import rasterio
from rasterio import warp

GeoTiffBands = TypedDict(
    "GeoTiffBands",
    {
        "red": np.ndarray,
        "green": np.ndarray,
        "blue": np.ndarray,
        "label": np.ndarray
    }
)
GeoTiffMetadata = TypedDict(
    "GeoTiffMetadata",
    {
        "width": int,
        "height": int,
        "latitude": float,
        "longitude": float,
        "bands": GeoTiffBands
    }
)

BAND_MAP = { "I2": 2, "I4": 4, "M11": 6, "LABEL": 7 }

def get_geotiff_metadata(image_path: str | Path) -> GeoTiffMetadata:
    """
    Extracts metadata (size, coordinates) from a GeoTIFF.
    """
    with rasterio.open(image_path) as dataset:
        bands: GeoTiffBands = {
            "red": dataset.read(BAND_MAP["I4"]),
            "green": dataset.read(BAND_MAP["I2"]),
            "blue": dataset.read(BAND_MAP["M11"]),
            "label": dataset.read(BAND_MAP["LABEL"]),
        }

        # Get bounds in the image's native CRS (Coordinate Reference System)
        bounds = dataset.bounds

        # Calculate center point in native CRS
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2

        # Transform to WGS84 (Latitude/Longitude) if not already
        # EPSG:4326 is the standard for Lat/Lon
        if dataset.crs != "EPSG:4326":
            longitude, latitude = warp.transform(
                src_crs=dataset.crs,
                dst_crs=rasterio.CRS.from_epsg("4326"),
                xs=[center_x],
                ys=[center_y]
            )
            return {
                "width": dataset.width,
                "height": dataset.height,
                "latitude": latitude[0],
                "longitude": longitude[0],
                "bands": bands
            }
        else:
            return {
                "width": dataset.width,
                "height": dataset.height,
                "latitude": center_y,
                "longitude": center_x,
                "bands": bands
            }


def _normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize the 16-bit / float band to 0-255 uint8, handling NaNs."""
    # Replace NaNs with 0 before any calculation
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)

    lower = np.percentile(band, 1)
    upper = np.percentile(band, 99)

    # Avoid division by zero if flat image
    if upper == lower:
        return np.zeros_like(band, dtype=np.uint8)

    # Clip to remove outliers
    band = np.clip(band, lower, upper)

    # Normalize and safe cast
    normalized = ((band - lower) / (upper - lower) * 255.0)
    return normalized.astype(np.uint8)


def _normalize_thermal_band(band: np.ndarray) -> np.ndarray:
    """
    Log-scale normalization specifically for the Fire Band (I4).
    Prevents the 'blob' effect by preserving gradients in high-heat areas.
    """
    # Replace NaNs
    band = np.nan_to_num(band, nan=0.0)

    # Log Transform: Compresses the huge range of fire intensities
    # We add 1.0 to avoid log(0)
    band_log = np.log1p(band - np.min(band))

    # Robust Scaling using Max instead of Percentile for the upper bound
    # This ensures the hottest pixel is 255, preserving the gradient.
    lower = np.percentile(band_log, 2)  # Clip bottom noise
    upper = np.max(band_log)  # NEVER clip the fire peak!

    if upper == lower:
        return np.zeros_like(band, dtype=np.uint8)

    normalized = ((band_log - lower) / (upper - lower) * 255.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def geotiff_to_jpg(image_path: str | Path) -> np.ndarray:
    """
    Converts a GeoTIFF to a JPEG image.
    """
    bands = get_geotiff_metadata(image_path=image_path)["bands"]

    # Create Composite
    return cv2.merge([
        _normalize_band(bands["blue"]),
        _normalize_band(bands["green"]),
        _normalize_thermal_band(bands["red"])
    ])

