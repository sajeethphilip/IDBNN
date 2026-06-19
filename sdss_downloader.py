#!/usr/bin/env python3
"""
SDSS Image Downloader for Galaxy Samples
Downloads FITS images from SDSS for ring and non-ring galaxies

Requirements:
    pip install astroquery astropy pandas pillow

Usage:
    python sdss_downloader.py                    # Test with 3 objects (FITS)
    python sdss_downloader.py --all              # Download all objects (FITS)
    python sdss_downloader.py --jpeg --all       # Download all as JPEG
    python sdss_downloader.py --jpeg --max 10    # Download 10 JPEGs per category
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import time
import re
from pathlib import Path

# Check for required packages
try:
    from astroquery.sdss import SDSS
except ImportError:
    print("ERROR: astroquery is required!")
    print("Install with: pip install astroquery astropy pandas")
    sys.exit(1)

# PIL is optional - only needed for JPEG conversion
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Configuration
DATA_DIR = "Data"
RINGS_DIR = os.path.join(DATA_DIR, "Galaxy", "Rings")
NONRINGS_DIR = os.path.join(DATA_DIR, "Galaxy", "NonRings")

os.makedirs(RINGS_DIR, exist_ok=True)
os.makedirs(NONRINGS_DIR, exist_ok=True)

def clean_filename(text):
    """Clean filename by removing invalid characters"""
    return re.sub(r'[^\w\-_.]', '_', str(text))

def format_coord_string(ra, dec):
    """Format coordinates for filename"""
    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    ra_str = coord.ra.to_string(u.hour, sep='', precision=2, pad=True)
    dec_str = coord.dec.to_string(u.deg, sep='', precision=2, pad=True, alwayssign=True)
    return f"{ra_str}_{dec_str}"

def fits_to_jpeg(fits_file, output_path):
    """Convert FITS file to JPEG"""
    if not HAS_PIL:
        return None

    try:
        # Read the FITS file
        hdul = fits.open(fits_file)

        # Get the data from the first HDU
        data = hdul[0].data

        # If data is None or empty, try the next HDU
        if data is None:
            for hdu in hdul[1:]:
                if hdu.data is not None:
                    data = hdu.data
                    break

        if data is None:
            hdul.close()
            return None

        # Handle multi-extension data
        if len(data.shape) == 3:
            # Take the first channel or average
            if data.shape[0] == 1:
                data = data[0]
            else:
                data = np.mean(data, axis=0)

        # Remove NaN and infinite values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to 8-bit with percentile clipping for better contrast
        p1, p99 = np.percentile(data, [1, 99])
        if p99 > p1:
            data_clipped = np.clip(data, p1, p99)
            data_norm = ((data_clipped - p1) / (p99 - p1) * 255).astype(np.uint8)
        else:
            data_norm = np.zeros_like(data, dtype=np.uint8)

        # Create and save JPEG
        if len(data_norm.shape) == 2:
            img = Image.fromarray(data_norm, mode='L')
        else:
            img = Image.fromarray(data_norm)

        img.save(output_path, quality=95)
        hdul.close()
        return output_path

    except Exception as e:
        print(f"    JPEG conversion error: {e}")
        return None

def download_sdss_image(ra, dec, output_dir, image_name="sdss_image",
                        size=256, retries=3, data_release=17, format_type='fits'):
    """
    Download SDSS image using astroquery
    """
    coord_string = format_coord_string(ra, dec)
    clean_name = clean_filename(str(image_name))
    ext = 'fits' if format_type == 'fits' else 'jpg'
    filename = f"{clean_name}_{coord_string}.{ext}"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"  ✓ File exists: {filename}")
        return output_path

    coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))

    for attempt in range(retries):
        try:
            print(f"  Downloading {format_type.upper()}: {image_name} at ({ra:.4f}, {dec:.4f}) [attempt {attempt+1}]")

            # Set SDSS Data Release
            SDSS.DR = data_release

            # Calculate radius in arcseconds (SDSS pixel scale ~0.396 arcsec/pixel)
            radius_arcsec = size * 0.396

            # Direct image cutout
            images = SDSS.get_images(
                coordinates=coord,
                radius=radius_arcsec * u.arcsec
            )

            if images and len(images) > 0:
                # Save the FITS file
                fits_path = output_path if format_type == 'fits' else output_path.replace('.jpg', '.fits')
                images[0].writeto(fits_path, overwrite=True)

                if format_type == 'fits':
                    print(f"  ✓ Saved FITS: {os.path.basename(fits_path)}")
                    return fits_path
                else:
                    # Convert to JPEG
                    jpeg_path = fits_to_jpeg(fits_path, output_path)
                    if jpeg_path:
                        print(f"  ✓ Saved JPEG: {os.path.basename(jpeg_path)}")
                        # Optionally keep the FITS file or delete it
                        # os.remove(fits_path)  # Uncomment to delete FITS after conversion
                        return jpeg_path
                    else:
                        print(f"  ✓ Saved FITS (JPEG conversion failed): {os.path.basename(fits_path)}")
                        return fits_path
            else:
                print(f"    No data found - trying with larger radius...")
                # Try with larger radius
                try:
                    images = SDSS.get_images(
                        coordinates=coord,
                        radius=(radius_arcsec * 2) * u.arcsec
                    )
                    if images and len(images) > 0:
                        fits_path = output_path if format_type == 'fits' else output_path.replace('.jpg', '.fits')
                        images[0].writeto(fits_path, overwrite=True)
                        print(f"  ✓ Saved FITS (larger cutout): {os.path.basename(fits_path)}")
                        return fits_path
                except:
                    pass

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                print(f"    Timeout, retrying...")
            else:
                print(f"    Error: {e}")
            time.sleep(2)

    print(f"  ✗ Failed: {filename}")
    return None

def process_csv_file(csv_path, output_dir, base_name, max_objects=None,
                     size=256, format_type='fits'):
    """Process a CSV file and download images for all entries"""
    print(f"\nProcessing: {csv_path}")
    print(f"  Format: {format_type.upper()}")
    print(f"  Size: {size}x{size} pixels")

    try:
        df = pd.read_csv(csv_path)
        print(f"  Read {len(df)} rows")
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return []

    # Identify RA and DEC columns
    ra_col = None
    dec_col = None

    for col in df.columns:
        col_upper = col.upper()
        if 'RA' in col_upper or '_RAJ2000' in col:
            ra_col = col
        if 'DEC' in col_upper or '_DEJ2000' in col:
            dec_col = col

    if ra_col is None or dec_col is None:
        print(f"  Error: Could not find RA/DEC columns")
        print(f"  Columns found: {df.columns.tolist()}")
        return []

    df_valid = df.dropna(subset=[ra_col, dec_col])

    if max_objects is not None and max_objects > 0:
        df_valid = df_valid.head(max_objects)

    print(f"  Processing {len(df_valid)} objects...")

    downloaded_files = []
    failed = 0
    start_time = time.time()

    for idx, row in df_valid.iterrows():
        try:
            ra = float(row[ra_col])
            dec = float(row[dec_col])

            # Get object name
            if 'rec' in row and pd.notna(row['rec']):
                obj_name = f"{base_name}_{int(row['rec'])}"
            elif 'SDSS_Obj' in row and pd.notna(row['SDSS_Obj']):
                obj_name = f"{base_name}_{str(row['SDSS_Obj'])}"
            else:
                obj_name = f"{base_name}_obj_{idx:04d}"

            filepath = download_sdss_image(ra, dec, output_dir, obj_name,
                                          size, 3, 17, format_type)

            if filepath:
                downloaded_files.append(filepath)
            else:
                failed += 1

            # Rate limiting
            time.sleep(0.5)

            # Progress update
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {idx+1}/{len(df_valid)} objects, {len(downloaded_files)} downloaded, {failed} failed, {elapsed:.1f}s elapsed")

        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            failed += 1
            continue

    elapsed = time.time() - start_time
    print(f"  Downloaded: {len(downloaded_files)} files in {elapsed:.1f}s")
    if failed > 0:
        print(f"  Failed: {failed} files")

    return downloaded_files

def main():
    parser = argparse.ArgumentParser(
        description='Download SDSS images using astroquery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Test with 3 objects (FITS)
  %(prog)s --all              # Download all objects (FITS)
  %(prog)s --jpeg --all       # Download all as JPEG (converts FITS to JPEG)
  %(prog)s --jpeg --max 10    # Download 10 JPEGs per category
  %(prog)s --size 512         # Download 512x512 pixel images
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Download all objects (default: test with 3)')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum objects to download per category')
    parser.add_argument('--size', type=int, default=256,
                        help=f'Image size in pixels (default: 256)')
    parser.add_argument('--jpeg', action='store_true',
                        help='Download JPEG instead of FITS (converts FITS to JPEG)')
    parser.add_argument('--rings-only', action='store_true',
                        help='Only download ring galaxies')
    parser.add_argument('--nonrings-only', action='store_true',
                        help='Only download non-ring galaxies')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in seconds for each request (default: 30)')

    args = parser.parse_args()

    # Determine max objects
    if args.all:
        max_objects = None
    elif args.max is not None:
        max_objects = args.max
    else:
        max_objects = 3  # Test mode by default

    format_type = 'jpeg' if args.jpeg else 'fits'

    print("="*70)
    print("SDSS Galaxy Image Downloader")
    print("="*70)
    print(f"Format: {format_type.upper()}")
    print(f"Image size: {args.size}x{args.size} pixels")
    print(f"Timeout: {args.timeout}s per request")
    if max_objects:
        print(f"Max objects per category: {max_objects}")
    else:
        print("Downloading ALL objects")

    if format_type == 'jpeg' and not HAS_PIL:
        print("WARNING: PIL not installed. Will save FITS files instead of JPEG.")
        print("Install PIL with: pip install pillow")
    print("="*70)

    rings_csv = "Images_Rings.csv"
    nonrings_csv = "NonRing_corr_list.csv"

    total_downloaded = 0

    # Download ring galaxies
    if os.path.exists(rings_csv) and not args.nonrings_only:
        print("\n" + "-"*60)
        print("RING GALAXIES")
        print("-"*60)
        downloaded = process_csv_file(rings_csv, RINGS_DIR, "Ring",
                                      max_objects, args.size, format_type)
        total_downloaded += len(downloaded)

    # Download non-ring galaxies
    if os.path.exists(nonrings_csv) and not args.rings_only:
        print("\n" + "-"*60)
        print("NON-RING GALAXIES")
        print("-"*60)
        downloaded = process_csv_file(nonrings_csv, NONRINGS_DIR, "NonRing",
                                      max_objects, args.size, format_type)
        total_downloaded += len(downloaded)

    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Format requested: {format_type.upper()}")
    print(f"Total files downloaded: {total_downloaded}")
    print(f"Rings: {RINGS_DIR}")
    print(f"NonRings: {NONRINGS_DIR}")

    if format_type == 'jpeg' and not HAS_PIL:
        print("\nNOTE: Files were saved as FITS because PIL (pillow) is not installed.")
        print("To get JPEG files, install pillow: pip install pillow")
    print("="*70)





if __name__ == "__main__":
    print('''
# Test with 3 objects (default)
python sdss_downloader.py

# Download all objects (FITS format)
python sdss_downloader.py --all

# Download 10 objects per category
python sdss_downloader.py --max 10

# Download JPEG instead of FITS
python sdss_downloader.py --jpeg --max 10

# Download all as JPEG with 512x512 size
python sdss_downloader.py --jpeg --all --size 512

# Download only ring galaxies
python sdss_downloader.py --rings-only --all

# Download only non-ring galaxies as JPEG
python sdss_downloader.py --nonrings-only --jpeg --all
    ''')
    main()
