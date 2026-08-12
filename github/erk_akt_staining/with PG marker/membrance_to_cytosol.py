import os
import cv2
import nd2
import numpy as np
import pandas as pd
import tifffile as tiff  # Requires: pip install tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology
import matplotlib.pyplot as plt

def run_dapi_cytosol_rings_pipeline(
    image_path,
    output_dir="membrane_vs_cytosol_analysis",
    dapi_ch=0,
    green_ch=3,
    red_ch=2,
    membrane_thickness=8,
    dapi_dilation=8,
):
    """Calculates frame-wide metrics for ND2 or TIFF files:
    
    Mean Green in Outer Membrane Ring / Mean Green in a clean DAPI-Dilated Cytosol Zone.
    """
    plots_dir = os.path.join(output_dir, "debug_plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary_records = []
    ext = os.path.splitext(image_path)[1].lower()

    # 1. Load data based on file extension (ND2 or TIFF)
    if ext in ['.tif', '.tiff']:
        print(f"Loaded TIFF file: {image_path}")
        with tiff.TiffFile(image_path) as tif:
            raw_data = tif.asarray()
            if len(raw_data.shape) == 3:
                raw_data = np.expand_dims(raw_data, axis=0)
            num_points = raw_data.shape[0]
            
    elif ext == '.nd2':
        print(f"Loaded ND2 file: {image_path}")
        with nd2.ND2File(image_path) as file:
            raw_data = file.asarray()
            num_points = file.sizes.get("P", 1)
            
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please use .nd2, .tif, or .tiff")

    def to_8bit(img):
        img_min, img_max = img.min(), img.max()
        if img_max - img_min == 0:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    # 2. Process each frame
    for p in range(num_points):
        print(f"Processing Frame P{p}...", end="", flush=True)

        # Extract raw physical data matrices
        raw_red = raw_data[p, red_ch, :, :]
        raw_green = raw_data[p, green_ch, :, :]
        raw_dapi = raw_data[p, dapi_ch, :, :]

        # A. Generate Solid Tissue ROI Mask (from Red Channel)
        img_red_8bit = to_8bit(raw_red)
        blurred_red = filters.gaussian(img_red_8bit, sigma=2.5)
        optimized_thresh = filters.threshold_otsu(blurred_red) * 0.45
        binary_borders = blurred_red > optimized_thresh
        closed_borders = morphology.closing(binary_borders, morphology.disk(4))
        solid_tissue_mask = ndi.binary_fill_holes(closed_borders)
        solid_tissue_mask = morphology.remove_small_objects(solid_tissue_mask, min_size=500)

        # B. Segment Clean Nuclei Mask (Your trusted logic using tissue values only)
        img_dapi_8bit = to_8bit(raw_dapi)
        cropped_dapi_8bit = cv2.bitwise_and(
            img_dapi_8bit, img_dapi_8bit, mask=solid_tissue_mask.astype(np.uint8)
        )
        blurred_dapi = filters.gaussian(cropped_dapi_8bit, sigma=1.5)
        
        thresh_dapi = filters.threshold_otsu(blurred_dapi[solid_tissue_mask > 0])
        binary_nuclei = (blurred_dapi > thresh_dapi) & (solid_tissue_mask > 0)
        binary_nuclei = morphology.remove_small_objects(binary_nuclei, min_size=300)
        nuclei_mask = ndi.binary_fill_holes(binary_nuclei)

        # =========================================================================
        # 3. Create Biology-Driven Concentric Sub-Regions
        # =========================================================================
        # Membrane: Outer edge of the solid tissue cluster
        tissue_eroded_mem = morphology.binary_erosion(
            solid_tissue_mask, morphology.disk(membrane_thickness)
        )
        membrane_mask = solid_tissue_mask & (~tissue_eroded_mem)

        # Cytosol: Dilated clean nuclei ring trapped strictly within tissue boundaries
        dilated_nuclei = morphology.binary_dilation(nuclei_mask, morphology.disk(dapi_dilation))
        raw_cytosol_mask = dilated_nuclei & (~nuclei_mask) & (solid_tissue_mask > 0)
        
        # Guard: Erase any parts of the cytosol mask that touch the outer membrane ring
        cytosol_mask = raw_cytosol_mask & (~membrane_mask)

        # =========================================================================
        # 4. Perform Symmetrical Quantification (Green Channel)
        # =========================================================================
        green_pixels_in_membrane = raw_green[membrane_mask]
        green_pixels_in_cytosol = raw_green[cytosol_mask]

        if len(green_pixels_in_membrane) > 0 and len(green_pixels_in_cytosol) > 0:
            # Symmetrical Mean: Clear geometric separation means no math tuning needed
            mean_green_membrane = np.quantile(green_pixels_in_membrane,0.85)
            mean_green_cytosol = np.quantile(green_pixels_in_cytosol,0.85)
            final_normalized_ratio = mean_green_membrane / mean_green_cytosol
        else:
            mean_green_membrane, mean_green_cytosol, final_normalized_ratio = 0, 0, np.nan

        summary_records.append({
            "Frame_P": p,
            "Membrane_Ring_Area_Pixels": np.sum(membrane_mask),
            "Cytosol_Zone_Area_Pixels": np.sum(cytosol_mask),
            "Mean_Green_Outer_Membrane": round(mean_green_membrane, 2),
            "Mean_Green_Inner_Cytosol": round(mean_green_cytosol, 2),
            "Normalized_Ratio_Membrane_Over_Cytosol": round(final_normalized_ratio, 4),
        })

        # =========================================================================
        # 5. Generate and Save Validated QC Plot Layout (3 Panels)
        # =========================================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Frame P{p} - Clean DAPI-Anchored QC (Ratio: {final_normalized_ratio:.4f})",
            fontsize=14, fontweight="bold"
        )

        # Panel 1: Outline of solid tissue mask over red channel
        contours_tissue, _ = cv2.findContours(
            solid_tissue_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        overlay_red = cv2.cvtColor(img_red_8bit, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay_red, contours_tissue, -1, (0, 255, 255), 3)  # Cyan
        axes[0].imshow(cv2.cvtColor(overlay_red, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Solid Tissue Border (Cyan)")

        # Panel 2: Outer Membrane Ring highlighted over Green Channel
        img_green_8bit = to_8bit(raw_green)
        overlay_membrane = cv2.cvtColor(img_green_8bit, cv2.COLOR_GRAY2BGR)
        overlay_membrane[membrane_mask] = [0, 255, 255]  # Cyan
        axes[1].imshow(cv2.cvtColor(overlay_membrane, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"2. Outer Membrane Ring ({membrane_thickness}px)")

        # Panel 3: Clean Cytosol Ring highlighted over Green Channel
        overlay_cytosol = cv2.cvtColor(img_green_8bit, cv2.COLOR_GRAY2BGR)
        overlay_cytosol[cytosol_mask] = [255, 0, 255]  # Magenta
        axes[2].imshow(cv2.cvtColor(overlay_cytosol, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"3. DAPI-Derived Cytosol (Dilation: {dapi_dilation}px)")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, f"qc_summary_P{p:03d}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(" Saved Plot.")

    # Export compiled spreadsheet data
    df_output = pd.DataFrame(summary_records)
    csv_filename = os.path.join(output_dir, "membrane_cytosol_dapi_summary.csv")
    df_output.to_csv(csv_filename, index=False)

    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUCCESSFUL")
    print(f"Master file saved to: {csv_filename}")
    print("=" * 60)
    print(df_output.to_string(index=False))





# --- Execute Target Call ---
light_file_path = r"\\10.64.139.2\data\Temp - Please delete your files after copying\billy\image_data\2026-5-28 20260320 retake\phm topto light 96hrs.nd2"
dark_file_path = r"\\10.64.139.2\data\Temp - Please delete your files after copying\billy\image_data\2026-5-28 20260320 retake\phm topto dark 96hrs.nd2"
run_dapi_cytosol_rings_pipeline(light_file_path, output_dir="optosos_light_pakt")
run_dapi_cytosol_rings_pipeline(dark_file_path, output_dir="optosos_dark_pakt")
