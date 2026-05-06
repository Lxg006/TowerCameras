"""
Build a LAS point cloud from a coordinate CSV and an optional RGB image.

Input:  camer_3dcoor_*.txt (columns: ScreenX, ScreenY, WorldX, WorldY, WorldZ)
        PNG/JPG (optional; used to sample RGB at ScreenX, ScreenY)
Output: point_cloud.las
"""

import numpy as np
import csv
import os

# ==================== Configuration ====================
# Input coordinate file path
input_file = r"C:\Users\Administrator\Desktop\0130\camer_3dcoor_20260409194303.txt"
# Source image path (used to sample RGB per screen pixel)
image_path = r"C:\Users\Administrator\Desktop\0130\0423.png"
# Output LAS file path
output_las_file = r"C:\Users\Administrator\Desktop\0130\point_cloud0423.las"

# ==================== Main program ====================


def load_image(image_path):
    """Load an image using PIL first, then fall back to OpenCV."""
    img_array = None
    image_format = None

    # Try PIL
    try:
        from PIL import Image
        img = Image.open(image_path)
        img_array = np.array(img)
        image_format = 'PIL'
        print(f"  Loaded image with PIL: {image_path}")
        return img_array, image_format
    except Exception as e:
        print(f"  PIL load failed: {e}")

    # Try OpenCV
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            # OpenCV uses BGR; convert to RGB
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_format = 'OpenCV'
            print(f"  Loaded image with OpenCV: {image_path}")
            return img_array, image_format
    except Exception as e:
        print(f"  OpenCV load failed: {e}")

    return None, None


def extract_rgb_from_image(img_array, screen_x, screen_y):
    """Sample RGB at integer (screen_x, screen_y) from the image array."""
    x = int(screen_x)
    y = int(screen_y)

    # Bounds check against image shape
    if 0 <= y < img_array.shape[0] and 0 <= x < img_array.shape[1]:
        if len(img_array.shape) == 3:
            # Multi-channel image
            if img_array.shape[2] >= 3:
                r = int(img_array[y, x, 0])
                g = int(img_array[y, x, 1])
                b = int(img_array[y, x, 2])
                return r, g, b
            else:
                # Single channel replicated to RGB
                gray_val = int(img_array[y, x, 0])
                return gray_val, gray_val, gray_val
        else:
            # 2D grayscale
            gray_val = int(img_array[y, x])
            return gray_val, gray_val, gray_val
    else:
        # Out of bounds: black
        return 0, 0, 0


def main():
    print("=" * 60)
    print("LAS point cloud generator")
    print("=" * 60)

    # Verify input file exists
    if not os.path.exists(input_file):
        print(f"Error: input file not found: {input_file}")
        return

    if not os.path.exists(image_path):
        print(f"Warning: image file not found: {image_path}")
        print("Will use elevation-based pseudo-color RGB instead.")
        use_image = False
    else:
        use_image = True

    # Load image if available
    img_array = None
    if use_image:
        img_array, image_format = load_image(image_path)
        if img_array is not None:
            print(f"  Image shape: {img_array.shape}")
        else:
            print("  Image load failed; will use elevation-based pseudo-color RGB.")
            use_image = False

    # Read coordinate rows
    print(f"\nReading coordinates: {input_file}")
    points = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Tip: install tqdm for a progress bar (pip install tqdm)")

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        # Header row
        try:
            header = next(reader)
            print(f"Header: {header}")
        except StopIteration:
            print("Warning: file is empty")
            return

        if use_tqdm:
            reader = tqdm(reader, desc="Reading rows", unit="row", dynamic_ncols=True)

        processed_count = 0
        error_count = 0

        for row_idx, row in enumerate(reader):
            if len(row) < 5:
                continue

            try:
                screen_x = float(row[0])
                screen_y = float(row[1])
                world_x = float(row[2])
                world_y = float(row[3])
                world_z = float(row[4])

                # RGB from image or placeholder (filled later from Z if no image)
                if use_image and img_array is not None:
                    r, g, b = extract_rgb_from_image(img_array, screen_x, screen_y)
                else:
                    r, g, b = 0, 0, 0  # replaced by Z-based coloring when no image

                points.append({
                    'screen_x': screen_x,
                    'screen_y': screen_y,
                    'world_x': world_x,
                    'world_y': world_y,
                    'world_z': world_z,
                    'r': r,
                    'g': g,
                    'b': b
                })

                processed_count += 1

                if not use_tqdm and processed_count % 10000 == 0:
                    print(f"  Processed {processed_count} rows...")

            except (ValueError, IndexError) as e:
                error_count += 1
                if not use_tqdm and error_count <= 5:  # only show first few bad rows
                    print(f"  Warning: skipped row {row_idx + 1} (invalid format)")
                continue

    if len(points) == 0:
        print("Error: no valid points")
        return

    print(f"\nRead complete:")
    print(f"  Valid points: {len(points)}")
    print(f"  Bad rows: {error_count}")

    # Write LAS
    print(f"\nWriting LAS...")
    try:
        import laspy

        # LAS 1.4, point format 3 includes RGB channels
        las = laspy.create(file_version="1.4", point_format=3)

        # World coordinates as arrays
        world_x_coords = np.array([p['world_x'] for p in points])
        world_y_coords = np.array([p['world_y'] for p in points])
        world_z_coords = np.array([p['world_z'] for p in points])

        las.x = world_x_coords
        las.y = world_y_coords
        las.z = world_z_coords

        # Intensity from normalized Z (world elevation)
        z_min = np.min(world_z_coords)
        z_max = np.max(world_z_coords)
        if z_max > z_min:
            intensity = ((world_z_coords - z_min) / (z_max - z_min) * 65535).astype(np.uint16)
        else:
            intensity = np.zeros(len(points), dtype=np.uint16)
        las.intensity = intensity

        # RGB channels
        if use_image and img_array is not None:
            # True color from image samples
            red_values = np.array([p['r'] for p in points], dtype=np.uint16)
            green_values = np.array([p['g'] for p in points], dtype=np.uint16)
            blue_values = np.array([p['b'] for p in points], dtype=np.uint16)

            # Scale 8-bit samples to LAS 16-bit color (0–255 → 0–65535)
            if red_values.max() <= 255:
                red_values = (red_values * 257).astype(np.uint16)  # 0-255 -> 0-65535
                green_values = (green_values * 257).astype(np.uint16)
                blue_values = (blue_values * 257).astype(np.uint16)

            las.red = red_values
            las.green = green_values
            las.blue = blue_values
            print("  Using RGB sampled from the image")
        else:
            # Pseudo-color from Z using viridis
            try:
                import matplotlib.pyplot as plt
                depth_normalized = (world_z_coords - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(world_z_coords)
                try:
                    viridis = plt.colormaps['viridis']
                except (AttributeError, KeyError):
                    import matplotlib.cm as cm
                    viridis = cm.get_cmap('viridis')
                rgb_colors = viridis(depth_normalized)[:, :3]
                las.red = (rgb_colors[:, 0] * 65535).astype(np.uint16)
                las.green = (rgb_colors[:, 1] * 65535).astype(np.uint16)
                las.blue = (rgb_colors[:, 2] * 65535).astype(np.uint16)
                print("  Using Z-based pseudo-color RGB (viridis)")
            except Exception:
                # Matplotlib missing: drive RGB from intensity (grayscale)
                gray_value = intensity.astype(np.uint16)
                las.red = gray_value
                las.green = gray_value
                las.blue = gray_value
                print("  Using intensity-based grayscale RGB")

        # Standard LAS fields (defaults for unclassified airborne-style data)
        las.classification = np.zeros(len(points), dtype=np.uint8)  # unclassified
        las.return_number = np.ones(len(points), dtype=np.uint8)
        las.number_of_returns = np.ones(len(points), dtype=np.uint8)
        las.scan_angle_rank = np.zeros(len(points), dtype=np.int8)

        # Header: minimum corner offsets and quantization scales (1 mm)
        las.header.offsets = [np.min(world_x_coords), np.min(world_y_coords), np.min(world_z_coords)]
        las.header.scales = [0.001, 0.001, 0.001]

        las.write(output_las_file)

        print(f"\nLAS written successfully.")
        print(f"  Output: {output_las_file}")
        print(f"  Point count: {len(points)}")
        print(f"  Coordinate bounds:")
        print(f"    X: {np.min(world_x_coords):.6f} ~ {np.max(world_x_coords):.6f}")
        print(f"    Y: {np.min(world_y_coords):.6f} ~ {np.max(world_y_coords):.6f}")
        print(f"    Z: {np.min(world_z_coords):.6f} ~ {np.max(world_z_coords):.6f}")
        print(f"  Z range: {z_min:.6f} ~ {z_max:.6f}")

    except ImportError:
        print("Error: laspy is not installed")
        print("Install with: pip install laspy")
    except Exception as e:
        print(f"Error: failed to write LAS: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
