import os
import numpy as np
import pandas as pd
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import tempfile
import subprocess
import sys
from difflib import SequenceMatcher
import re

class ImageMosaicCreator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Mosaic to PDF Creator")
        self.root.geometry("750x650")

        # Variables
        self.list_file = tk.StringVar()
        self.folder1 = tk.StringVar()
        self.folder2 = tk.StringVar()
        self.output_pdf = tk.StringVar(value="mosaics.pdf")
        self.pairs_per_row = tk.IntVar(value=2)  # Default 2 pairs per row
        self.pairs_per_page = tk.IntVar(value=2)  # Default 2 pairs per page (2x2 grid)
        self.matching_method = tk.StringVar(value="coordinates")  # 'coordinates' or 'fuzzy'
        self.coordinate_rounding = tk.IntVar(value=2)  # Round coordinates to N decimal places

        self.current_pdf_path = None
        self.create_widgets()

    def create_widgets(self):
        # Main frame with scrolling for better organization
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        # List file selection
        ttk.Label(file_frame, text="1. CSV list file:").grid(row=0, column=0, sticky='w', pady=5)
        frame1 = ttk.Frame(file_frame)
        frame1.grid(row=0, column=1, columnspan=2, sticky='ew', pady=5)
        ttk.Entry(frame1, textvariable=self.list_file, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(frame1, text="Browse", command=self.browse_list_file).pack(side=tk.LEFT)

        # Folder 1 selection
        ttk.Label(file_frame, text="2. First image folder:").grid(row=1, column=0, sticky='w', pady=5)
        frame2 = ttk.Frame(file_frame)
        frame2.grid(row=1, column=1, columnspan=2, sticky='ew', pady=5)
        ttk.Entry(frame2, textvariable=self.folder1, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(frame2, text="Browse", command=self.browse_folder1).pack(side=tk.LEFT)

        # Folder 2 selection
        ttk.Label(file_frame, text="3. Second image folder:").grid(row=2, column=0, sticky='w', pady=5)
        frame3 = ttk.Frame(file_frame)
        frame3.grid(row=2, column=1, columnspan=2, sticky='ew', pady=5)
        ttk.Entry(frame3, textvariable=self.folder2, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(frame3, text="Browse", command=self.browse_folder2).pack(side=tk.LEFT)

        # Output PDF
        ttk.Label(file_frame, text="4. Output PDF file:").grid(row=3, column=0, sticky='w', pady=5)
        frame4 = ttk.Frame(file_frame)
        frame4.grid(row=3, column=1, columnspan=2, sticky='ew', pady=5)
        ttk.Entry(frame4, textvariable=self.output_pdf, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(frame4, text="Browse", command=self.browse_output).pack(side=tk.LEFT)

        # Matching options frame
        matching_frame = ttk.LabelFrame(main_frame, text="Matching Options", padding=10)
        matching_frame.pack(fill=tk.X, pady=5)

        # Matching method selection
        ttk.Label(matching_frame, text="Matching method:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Radiobutton(matching_frame, text="Coordinate-based (RA/Dec)",
                       variable=self.matching_method, value="coordinates").grid(row=0, column=1, sticky='w', padx=5)
        ttk.Radiobutton(matching_frame, text="Fuzzy filename matching",
                       variable=self.matching_method, value="fuzzy").grid(row=0, column=2, sticky='w', padx=5)
        ttk.Radiobutton(matching_frame, text="Hybrid (coordinates first, then fuzzy)",
                       variable=self.matching_method, value="hybrid").grid(row=0, column=3, sticky='w', padx=5)

        # Coordinate rounding
        ttk.Label(matching_frame, text="Coordinate rounding (decimal places):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Spinbox(matching_frame, from_=0, to=6, textvariable=self.coordinate_rounding,
                   width=8).grid(row=1, column=1, sticky='w', padx=5)
        ttk.Label(matching_frame, text="(Higher = more precise matching)").grid(row=1, column=2, sticky='w', padx=5)

        # Similarity threshold
        ttk.Label(matching_frame, text="Fuzzy matching threshold:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.similarity_threshold = tk.DoubleVar(value=0.6)
        ttk.Scale(matching_frame, from_=0.0, to=1.0, variable=self.similarity_threshold,
                 orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, sticky='w', padx=5)
        self.threshold_label = ttk.Label(matching_frame, text="0.60")
        self.threshold_label.grid(row=2, column=2, sticky='w', padx=5)

        # Update threshold label
        def update_threshold_label(*args):
            self.threshold_label.config(text=f"{self.similarity_threshold.get():.2f}")
        self.similarity_threshold.trace('w', update_threshold_label)

        # Layout options frame
        layout_frame = ttk.LabelFrame(main_frame, text="Layout Options", padding=10)
        layout_frame.pack(fill=tk.X, pady=5)

        # Pairs per row control
        ttk.Label(layout_frame, text="Pairs per row:").grid(row=0, column=0, sticky='w', padx=5)
        self.pairs_per_row_spinbox = ttk.Spinbox(layout_frame, from_=1, to=5,
                                                  textvariable=self.pairs_per_row,
                                                  width=10, command=self.update_pairs_per_page)
        self.pairs_per_row_spinbox.grid(row=0, column=1, padx=5)
        ttk.Label(layout_frame, text="(Number of target comparisons per row)").grid(row=0, column=2, sticky='w', padx=5)

        # Pairs per page display (auto-calculated)
        ttk.Label(layout_frame, text="Pairs per page:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.pairs_per_page_label = ttk.Label(layout_frame, text="4", font=('Arial', 10, 'bold'))
        self.pairs_per_page_label.grid(row=1, column=1, sticky='w', padx=5)
        ttk.Label(layout_frame, text="(rows × columns)").grid(row=1, column=2, sticky='w', padx=5)

        # Button to add more rows
        ttk.Button(layout_frame, text="Add Row", command=self.add_row).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(layout_frame, text="Remove Row", command=self.remove_row).grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(layout_frame, text="(Add or remove rows to increase/decrease pairs per page)").grid(row=2, column=2, sticky='w', padx=5)

        # Page orientation
        ttk.Label(layout_frame, text="Page orientation:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.orientation = tk.StringVar(value="landscape")
        ttk.Radiobutton(layout_frame, text="Landscape", variable=self.orientation,
                       value="landscape").grid(row=3, column=1, sticky='w', padx=5)
        ttk.Radiobutton(layout_frame, text="Portrait", variable=self.orientation,
                       value="portrait").grid(row=3, column=2, sticky='w', padx=5)

        # Image options frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Options", padding=10)
        image_frame.pack(fill=tk.X, pady=5)

        # Image size
        ttk.Label(image_frame, text="Image size:").grid(row=0, column=0, sticky='w', padx=5)
        self.img_width = tk.IntVar(value=250)
        self.img_height = tk.IntVar(value=250)
        ttk.Label(image_frame, text="Width:").grid(row=0, column=1, padx=5)
        ttk.Spinbox(image_frame, from_=100, to=800, textvariable=self.img_width, width=8).grid(row=0, column=2, padx=5)
        ttk.Label(image_frame, text="Height:").grid(row=0, column=3, padx=5)
        ttk.Spinbox(image_frame, from_=100, to=800, textvariable=self.img_height, width=8).grid(row=0, column=4, padx=5)

        # Display options
        self.show_coords = tk.BooleanVar(value=True)
        ttk.Checkbutton(image_frame, text="Show coordinates on images",
                       variable=self.show_coords).grid(row=1, column=0, columnspan=2, pady=5, sticky='w', padx=5)

        self.show_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(image_frame, text="Show folder labels",
                       variable=self.show_labels).grid(row=1, column=2, columnspan=2, pady=5, sticky='w', padx=5)

        self.show_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(image_frame, text="Show grid lines",
                       variable=self.show_grid).grid(row=2, column=0, columnspan=2, pady=5, sticky='w', padx=5)

        self.show_similarity = tk.BooleanVar(value=True)
        ttk.Checkbutton(image_frame, text="Show similarity score",
                       variable=self.show_similarity).grid(row=2, column=2, columnspan=2, pady=5, sticky='w', padx=5)

        # Progress and status frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(pady=5, fill=tk.X)

        # Status label
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(pady=5)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Create button
        self.start_button = ttk.Button(button_frame, text="Create PDF", command=self.start_creation)
        self.start_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # View PDF button
        self.view_button = ttk.Button(button_frame, text="View PDF", command=self.view_pdf, state='disabled')
        self.view_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Configure grid weights
        file_frame.columnconfigure(1, weight=1)

    def update_pairs_per_page(self):
        """Update the pairs per page calculation"""
        try:
            rows = self.pairs_per_row.get()
            # For a grid layout, pairs per page = rows * 2 (since we always have 2 columns: folder1 and folder2)
            pairs_per_page = rows * 2
            self.pairs_per_page.set(pairs_per_page)
            self.pairs_per_page_label.config(text=str(pairs_per_page))
        except:
            pass

    def add_row(self):
        """Add a row to increase pairs per page"""
        current = self.pairs_per_row.get()
        if current < 5:
            self.pairs_per_row.set(current + 1)
            self.update_pairs_per_page()

    def remove_row(self):
        """Remove a row to decrease pairs per page"""
        current = self.pairs_per_row.get()
        if current > 1:
            self.pairs_per_row.set(current - 1)
            self.update_pairs_per_page()

    def browse_list_file(self):
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.list_file.set(filename)

    def browse_folder1(self):
        folder = filedialog.askdirectory(title="Select first image folder")
        if folder:
            self.folder1.set(folder)

    def browse_folder2(self):
        folder = filedialog.askdirectory(title="Select second image folder")
        if folder:
            self.folder2.set(folder)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save PDF as",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.output_pdf.set(filename)

    def parse_coordinates(self, filename):
        """Extract RA and Dec from filename with optional rounding"""
        try:
            # Remove extension
            name = filename.replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
            # Split by underscore
            parts = name.split('_')

            # Try to find coordinates in the filename
            coord_pattern = r'([-]?\d+\.\d+)[_-]([-]?\d+\.\d+)'
            match = re.search(coord_pattern, name)

            if match:
                ra = float(match.group(1))
                dec = float(match.group(2))

                # Apply rounding if specified
                rounding = self.coordinate_rounding.get()
                if rounding >= 0:
                    ra = round(ra, rounding)
                    dec = round(dec, rounding)

                return ra, dec

            # Fallback to last two parts
            if len(parts) >= 2:
                try:
                    ra = float(parts[-2])
                    dec = float(parts[-1])

                    # Apply rounding
                    rounding = self.coordinate_rounding.get()
                    if rounding >= 0:
                        ra = round(ra, rounding)
                        dec = round(dec, rounding)

                    return ra, dec
                except ValueError:
                    pass

        except (ValueError, IndexError):
            pass
        return None, None

    def calculate_coordinate_distance(self, ra1, dec1, ra2, dec2):
        """Calculate angular distance between two coordinates"""
        return math.sqrt((ra1 - ra2)**2 + (dec1 - dec2)**2)

    def calculate_filename_similarity(self, name1, name2):
        """Calculate similarity between two filenames"""
        # Remove extensions and convert to lowercase
        name1_clean = Path(name1).stem.lower()
        name2_clean = Path(name2).stem.lower()

        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, name1_clean, name2_clean).ratio()

        # Also check for partial matches (e.g., one contains the other)
        if name1_clean in name2_clean or name2_clean in name1_clean:
            similarity = max(similarity, 0.8)

        return similarity

    def find_best_match_coordinates(self, target_ra, target_dec, image_folder):
        """Find best match using coordinate-based matching"""
        best_match = None
        best_distance = float('inf')
        best_coords = None

        if not os.path.exists(image_folder):
            return None, None, None

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                ra, dec = self.parse_coordinates(filename)
                if ra is not None and dec is not None:
                    distance = self.calculate_coordinate_distance(target_ra, target_dec, ra, dec)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = filename
                        best_coords = (ra, dec)

        return best_match, best_distance, best_coords

    def find_best_match_fuzzy(self, target_name, image_folder):
        """Find best match using fuzzy filename matching"""
        best_match = None
        best_similarity = 0.0

        if not os.path.exists(image_folder):
            return None, 0.0

        target_clean = Path(target_name).stem.lower()

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                similarity = self.calculate_filename_similarity(target_clean, filename)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = filename

        # Only return if similarity meets threshold
        if best_similarity >= self.similarity_threshold.get():
            return best_match, best_similarity
        else:
            return None, best_similarity

    def find_best_match_hybrid(self, target_ra, target_dec, target_name, image_folder):
        """Hybrid matching: try coordinates first, fall back to fuzzy"""
        # Try coordinate matching first
        coord_match, coord_distance, coord_coords = self.find_best_match_coordinates(
            target_ra, target_dec, image_folder)

        # Try fuzzy matching
        fuzzy_match, fuzzy_similarity = self.find_best_match_fuzzy(target_name, image_folder)

        # Decide which to use
        if coord_match and coord_distance < 1.0:  # Close coordinate match
            return coord_match, coord_distance, coord_coords, 'coordinates'
        elif fuzzy_match and fuzzy_similarity >= self.similarity_threshold.get():
            # Parse coordinates for fuzzy match if available
            fuzzy_ra, fuzzy_dec = self.parse_coordinates(fuzzy_match)
            return fuzzy_match, fuzzy_similarity, (fuzzy_ra, fuzzy_dec), 'fuzzy'
        elif coord_match:
            return coord_match, coord_distance, coord_coords, 'coordinates'
        else:
            return fuzzy_match, fuzzy_similarity, None, 'fuzzy' if fuzzy_match else 'none'

    def find_best_match(self, target_file, image_folder):
        """Main matching function that uses the selected method"""
        target_name = Path(target_file).stem
        target_ra, target_dec = self.parse_coordinates(target_file)

        method = self.matching_method.get()

        if method == 'coordinates':
            if target_ra is not None and target_dec is not None:
                match, distance, coords = self.find_best_match_coordinates(
                    target_ra, target_dec, image_folder)
                return match, distance, coords, 'coordinates'
            else:
                return None, None, None, 'none'

        elif method == 'fuzzy':
            match, similarity = self.find_best_match_fuzzy(target_name, image_folder)
            if match:
                coords = self.parse_coordinates(match)
                return match, similarity, coords, 'fuzzy'
            else:
                return None, None, None, 'none'

        elif method == 'hybrid':
            if target_ra is not None and target_dec is not None:
                return self.find_best_match_hybrid(target_ra, target_dec, target_name, image_folder)
            else:
                match, similarity = self.find_best_match_fuzzy(target_name, image_folder)
                if match:
                    coords = self.parse_coordinates(match)
                    return match, similarity, coords, 'fuzzy'
                else:
                    return None, None, None, 'none'

        return None, None, None, 'none'

    def load_and_resize_image(self, image_path, target_size):
        """Load an image and resize it to target size"""
        try:
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize image maintaining aspect ratio
                img.thumbnail(target_size, Image.Resampling.LANCZOS)

                # Create a new image with the exact target size and paste the resized image centered
                new_img = Image.new('RGB', target_size, color='black')
                x_offset = (target_size[0] - img.size[0]) // 2
                y_offset = (target_size[1] - img.size[1]) // 2
                new_img.paste(img, (x_offset, y_offset))
                return new_img
            else:
                # Create a blank gray image with text
                img = Image.new('RGB', target_size, color='gray')
                draw = ImageDraw.Draw(img)
                # Try to add text
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

                text = "No Image Found"
                # Center the text
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (target_size[0] - text_width) // 2
                y = (target_size[1] - text_height) // 2
                draw.text((x, y), text, fill='white', font=font)
                return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', target_size, color='darkgray')

    def add_info_to_image(self, image, ra, dec, folder_name, target_name=None,
                          match_method=None, match_score=None):
        """Add coordinate and label information to the image"""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        lines = []
        if self.show_labels.get():
            lines.append(folder_name)
        if self.show_coords.get() and ra is not None and dec is not None:
            lines.append(f"RA: {ra:.3f}, Dec: {dec:.3f}")
        if target_name:
            lines.append(f"Target: {target_name[:20]}")  # Truncate long names
        if self.show_similarity.get() and match_score is not None:
            if match_method == 'coordinates':
                lines.append(f"Dist: {match_score:.4f}")
            elif match_method == 'fuzzy':
                lines.append(f"Similarity: {match_score:.2%}")

        if lines:
            text = "\n".join(lines)
            # Draw background rectangle
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([(5, 5), (15 + text_width, 15 + text_height)], fill='black')
            draw.text((10, 10), text, fill='white', font=font)

        return image

    def create_mosaic_image(self, images_grid, target_size, rows, cols):
        """Create a mosaic grid from a 2D list of image info"""
        try:
            # Load all images
            loaded_images = []
            for row in images_grid:
                row_images = []
                for img_info in row:
                    img = self.load_and_resize_image(img_info['path'], target_size)
                    img = self.add_info_to_image(img, img_info.get('ra'), img_info.get('dec'),
                                                img_info.get('folder', 'Unknown'),
                                                img_info.get('target'),
                                                img_info.get('match_method'),
                                                img_info.get('match_score'))
                    row_images.append(img)
                loaded_images.append(row_images)

            # Calculate mosaic dimensions
            mosaic_width = target_size[0] * cols
            mosaic_height = target_size[1] * rows
            mosaic = Image.new('RGB', (mosaic_width, mosaic_height), color='white')

            # Paste images
            for row_idx, row in enumerate(loaded_images):
                for col_idx, img in enumerate(row):
                    x_offset = col_idx * target_size[0]
                    y_offset = row_idx * target_size[1]
                    mosaic.paste(img, (x_offset, y_offset))

            # Add grid lines if enabled
            if self.show_grid.get():
                draw = ImageDraw.Draw(mosaic)
                line_color = 'gray'
                line_width = 2

                # Draw vertical grid lines
                for col in range(1, cols):
                    x = col * target_size[0]
                    draw.line([(x, 0), (x, mosaic_height)], fill=line_color, width=line_width)

                # Draw horizontal grid lines
                for row in range(1, rows):
                    y = row * target_size[1]
                    draw.line([(0, y), (mosaic_width, y)], fill=line_color, width=line_width)

                # Draw border
                draw.rectangle([(0, 0), (mosaic_width-1, mosaic_height-1)],
                              outline=line_color, width=line_width)

            return mosaic

        except Exception as e:
            print(f"Error creating mosaic: {e}")
            return None

    def view_pdf(self):
        """Open the generated PDF with the system's default PDF viewer"""
        if self.current_pdf_path and os.path.exists(self.current_pdf_path):
            try:
                if sys.platform == 'win32':
                    os.startfile(self.current_pdf_path)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', self.current_pdf_path])
                else:  # Linux
                    subprocess.run(['xdg-open', self.current_pdf_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open PDF:\n{str(e)}")
        else:
            messagebox.showwarning("Warning", "No PDF has been created yet. Please create a PDF first.")

    def create_pdf(self):
        """Main function to create PDF with mosaics"""
        temp_images = []

        try:
            # Read CSV file
            df = pd.read_csv(self.list_file.get())

            # Get list of target filenames (assuming first column contains filenames)
            target_files = df.iloc[:, 0].tolist()

            target_size = (self.img_width.get(), self.img_height.get())
            rows = self.pairs_per_row.get()
            cols = 2  # Always 2 columns (Folder 1 and Folder 2)
            pairs_per_page = rows * cols

            # Determine page size based on orientation
            if self.orientation.get() == "landscape":
                page_size = landscape(letter)
                page_width, page_height = page_size
            else:
                page_size = letter
                page_width, page_height = page_size

            # Calculate mosaic size (will be scaled to fit page with margins)
            mosaic_width_pixels = target_size[0] * cols
            mosaic_height_pixels = target_size[1] * rows

            # Calculate scale to fit page with margins
            margin = 72  # 1 inch in points
            max_width_points = page_width - 2 * margin
            max_height_points = page_height - 2 * margin - 50

            # Convert pixels to points (assuming 72 DPI for conversion)
            mosaic_width_points = mosaic_width_pixels
            mosaic_height_points = mosaic_height_pixels

            # Calculate scale factor
            scale_x = max_width_points / mosaic_width_points
            scale_y = max_height_points / mosaic_height_points
            scale = min(scale_x, scale_y, 1.0)

            # Calculate final mosaic size in points
            final_width = mosaic_width_points * scale
            final_height = mosaic_height_points * scale

            # Calculate position to center the mosaic
            x_position = (page_width - final_width) / 2
            y_position = (page_height - final_height) / 2 + 25

            # Process targets in batches
            total_pages = (len(target_files) + pairs_per_page - 1) // pairs_per_page
            self.progress['maximum'] = total_pages
            self.progress['value'] = 0
            self.progress.configure(mode='determinate')

            # Create PDF
            pdf_path = self.output_pdf.get()
            c = canvas.Canvas(pdf_path, pagesize=page_size)

            page_num = 0
            for page_idx in range(total_pages):
                start_idx = page_idx * pairs_per_page
                end_idx = min(start_idx + pairs_per_page, len(target_files))

                # Create a grid for this page
                images_grid = []
                current_row = []

                for idx in range(start_idx, end_idx):
                    target_file = target_files[idx]
                    target_name = Path(target_file).stem

                    # Find best matches for folder1 and folder2
                    match1, score1, coords1, method1 = self.find_best_match(target_file, self.folder1.get())
                    match2, score2, coords2, method2 = self.find_best_match(target_file, self.folder2.get())

                    folder1_info = {
                        'path': os.path.join(self.folder1.get(), match1) if match1 else None,
                        'ra': coords1[0] if coords1 else None,
                        'dec': coords1[1] if coords1 else None,
                        'folder': os.path.basename(self.folder1.get()),
                        'target': target_name,
                        'match_method': method1 if match1 else None,
                        'match_score': score1 if match1 else None
                    }

                    folder2_info = {
                        'path': os.path.join(self.folder2.get(), match2) if match2 else None,
                        'ra': coords2[0] if coords2 else None,
                        'dec': coords2[1] if coords2 else None,
                        'folder': os.path.basename(self.folder2.get()),
                        'target': target_name,
                        'match_method': method2 if match2 else None,
                        'match_score': score2 if match2 else None
                    }

                    # Add to current row
                    current_row.append(folder1_info)
                    current_row.append(folder2_info)

                    # If row is complete, add to grid
                    if len(current_row) >= cols or idx == end_idx - 1:
                        images_grid.append(current_row)
                        current_row = []

                # Create mosaic for this page
                mosaic_image = self.create_mosaic_image(images_grid, target_size, len(images_grid), cols)

                if mosaic_image:
                    # Save to temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    mosaic_image.save(temp_file.name, 'PNG')
                    temp_images.append(temp_file.name)
                    temp_file.close()

                    # Add to PDF
                    c.drawImage(temp_file.name, x_position, y_position,
                               width=final_width, height=final_height)

                    # Add title
                    c.setFont("Helvetica-Bold", 14)
                    title = f"Image Comparison - Page {page_num + 1}"
                    title_width = c.stringWidth(title, "Helvetica-Bold", 14)
                    c.drawString((page_width - title_width) / 2, page_height - 40, title)

                    # Add subtitle with matching method info
                    c.setFont("Helvetica", 10)
                    method_text = f"Matching method: {self.matching_method.get().title()}"
                    if self.matching_method.get() == 'coordinates':
                        method_text += f" (rounded to {self.coordinate_rounding.get()} decimals)"
                    elif self.matching_method.get() == 'fuzzy':
                        method_text += f" (threshold: {self.similarity_threshold.get():.2f})"

                    method_width = c.stringWidth(method_text, "Helvetica", 10)
                    c.drawString((page_width - method_width) / 2, page_height - 60, method_text)

                    # Add page number
                    page_num += 1
                    self.add_page_number(c, page_num, total_pages, page_width, page_height)

                    c.showPage()

                    # Update status
                    self.status_label.config(text=f"Creating page {page_num} of {total_pages}")

                self.progress['value'] = page_idx + 1
                self.root.update_idletasks()

            # Save PDF
            c.save()
            self.current_pdf_path = pdf_path

            # Enable view button
            self.view_button.config(state='normal')

            # Ask if user wants to view the PDF
            if messagebox.askyesno("Success",
                                  f"PDF created successfully!\n\n"
                                  f"Saved to: {pdf_path}\n"
                                  f"Total pages: {page_num}\n"
                                  f"Total comparisons: {len(target_files)}\n"
                                  f"Layout: {rows} rows × {cols} columns\n"
                                  f"Matching method: {self.matching_method.get().title()}\n\n"
                                  f"Would you like to view the PDF now?"):
                self.view_pdf()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temporary files
            for temp_file in temp_images:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            self.progress['value'] = 0
            self.progress.configure(mode='indeterminate')
            self.status_label.config(text="Ready")
            self.start_button.config(state='normal')

    def add_page_number(self, pdf_canvas, page_num, total_pages, width, height):
        """Add page number to PDF page"""
        pdf_canvas.saveState()
        pdf_canvas.setFont("Helvetica", 10)
        text = f"Page {page_num} of {total_pages}"
        text_width = pdf_canvas.stringWidth(text, "Helvetica", 10)
        pdf_canvas.drawString((width - text_width) / 2, 20, text)
        pdf_canvas.restoreState()

    def start_creation(self):
        """Start the PDF creation process in a separate thread"""
        # Validate inputs
        if not self.list_file.get():
            messagebox.showwarning("Warning", "Please select a list file")
            return
        if not self.folder1.get():
            messagebox.showwarning("Warning", "Please select first image folder")
            return
        if not self.folder2.get():
            messagebox.showwarning("Warning", "Please select second image folder")
            return
        if not self.output_pdf.get():
            messagebox.showwarning("Warning", "Please specify output PDF file")
            return

        self.update_pairs_per_page()
        self.progress.start()
        self.start_button.config(state='disabled')
        self.view_button.config(state='disabled')
        self.status_label.config(text="Creating PDF...")

        # Run in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self.create_pdf)
        thread.daemon = True
        thread.start()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageMosaicCreator()
    app.run()
