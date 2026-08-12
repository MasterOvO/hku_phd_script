import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, morphology
import nd2
import tifffile  # Ensure you have this installed: pip install tifffile

def run_nuclei_normalized_pipeline(
    image_path, output_dir="nuclei_normalized_analysis", red_ch=2, green_ch=1, dapi_ch=0
):
    """Calculates frame-wide metrics:
    Mean Raw Green in Tissue ROI / Mean Raw DAPI in Nuclei Only.
    Supports both .nd2 and .tiff/.tif files.
    """
    plots_dir = os.path.join(output_dir, "debug_plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary_records = []
    ext = os.path.splitext(image_path)[1].lower()

    # 1. Load data based on file extension
    if ext == ".nd2":
        with nd2.ND2File(image_path) as file:
            print(f"Loaded ND2 file: {image_path}")
            raw_data = file.asarray()
            num_points = file.sizes.get("P", 1)
    elif ext in [".tiff", ".tif"]:
        with tifffile.TiffFile(image_path) as file:
            print(f"Loaded TIFF file: {image_path}")
            raw_data = file.asarray()
            # For multi-position TIFFs, assuming standard shape: (Positions, Channels, Y, X)
            # If your TIFF is single-position (Channels, Y, X), expand dimensions to match
            if raw_data.ndim == 3:
                raw_data = np.expand_dims(raw_data, axis=0)
            num_points = raw_data.shape[0]
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please use .nd2 or .tiff/.tif")

    def to_8bit(img):
        img_min, img_max = img.min(), img.max()
        if img_max - img_min == 0:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    for p in range(num_points):
        print(f"Processing Frame P{p}...", end="", flush=True)

        # Extract raw physical data matrices
        raw_red = raw_data[p, red_ch, :, :]
        raw_green = raw_data[p, green_ch, :, :]
        raw_dapi = raw_data[p, dapi_ch, :, :]

        # 2. Generate Solid Tissue ROI Mask (from Red Channel)
        img_red_8bit = to_8bit(raw_red)
        blurred_red = filters.gaussian(img_red_8bit, sigma=2.5)
        optimized_thresh = filters.threshold_otsu(blurred_red) * 0.45
        binary_borders = blurred_red > optimized_thresh
        closed_borders = morphology.closing(binary_borders, morphology.disk(4))
        solid_tissue_mask = ndi.binary_fill_holes(closed_borders)
        solid_tissue_mask = morphology.remove_small_objects(solid_tissue_mask, min_size=500)

        # 3. Segment Clean Nuclei Mask (from DAPI Channel inside Tissue Mask)
        img_dapi_8bit = to_8bit(raw_dapi)
        cropped_dapi_8bit = cv2.bitwise_and(
            img_dapi_8bit, img_dapi_8bit, mask=solid_tissue_mask.astype(np.uint8)
        )
        blurred_dapi = filters.gaussian(cropped_dapi_8bit, sigma=1.5)

        # Threshold DAPI using only tissue values to avoid background noise
        thresh_dapi = filters.threshold_otsu(blurred_dapi[solid_tissue_mask > 0])
        binary_nuclei = (blurred_dapi > thresh_dapi) & (solid_tissue_mask > 0)

        # Remove noise artifacts (Objects smaller than 300 pixels)
        binary_nuclei = morphology.remove_small_objects(binary_nuclei, min_size=300)
        nuclei_mask = ndi.binary_fill_holes(binary_nuclei)

        # 4. Perform Target Quantification using Exact Masks
        green_pixels_in_tissue = raw_green[solid_tissue_mask]
        dapi_pixels_in_nuclei = raw_dapi[nuclei_mask]

        if len(green_pixels_in_tissue) > 0 and len(dapi_pixels_in_nuclei) > 0:
            mean_green_tissue = np.mean(green_pixels_in_tissue)
            mean_dapi_nuclei = np.mean(dapi_pixels_in_nuclei)
            final_normalized_ratio = mean_green_tissue / mean_dapi_nuclei
        else:
            mean_green_tissue, mean_dapi_nuclei, final_normalized_ratio = 0, 0, np.nan

        summary_records.append(
            {
                "Frame_P": p,
                "Tissue_Area_Pixels": np.sum(solid_tissue_mask),
                "Nuclei_Area_Pixels": np.sum(nuclei_mask),
                "Mean_Green_Tissue_ROI": round(mean_green_tissue, 2),
                "Mean_DAPI_Nuclei_Only": round(mean_dapi_nuclei, 2),
                "Normalized_Ratio_GreenROI_Over_DapiNuclei": round(final_normalized_ratio, 4),
            }
        )

        # 5. Generate and Save QC Plot Layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Frame P{p} - Nuclei-Normalized QC (Ratio: {final_normalized_ratio:.4f})",
            fontsize=14,
            fontweight="bold",
        )

        # Panel 1: Outline of tissue mask over red channel
        contours_tissue, _ = cv2.findContours(
            solid_tissue_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        overlay_red = cv2.cvtColor(img_red_8bit, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay_red, contours_tissue, -1, (0, 255, 255), 3)  # Cyan
        axes[0].imshow(cv2.cvtColor(overlay_red, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Solid Tissue Border (Cyan)")

        # Panel 2: Nuclei Mask Overlay over DAPI channel
        contours_nuc, _ = cv2.findContours(
            nuclei_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        overlay_dapi = cv2.cvtColor(img_dapi_8bit, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay_dapi, contours_nuc, -1, (255, 0, 0), 2)  # Red outlines for nuclei
        axes[1].imshow(cv2.cvtColor(overlay_dapi, cv2.COLOR_BGR2RGB))
        axes[1].set_title("2. Isolated Nuclei Boundaries (Red)")

        # Panel 3: Target Green channel visualization
        cropped_green = cv2.bitwise_and(
            to_8bit(raw_green),
            to_8bit(raw_green),
            mask=solid_tissue_mask.astype(np.uint8),
        )
        axes[2].imshow(cropped_green, cmap="gray")
        axes[2].set_title("3. Evaluated Green Intensity (Tissue)")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plot_filename = os.path.join(plots_dir, f"qc_summary_P{p:03d}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(" Saved Plot.")

    # Export compiled spreadsheet data
    df_output = pd.DataFrame(summary_records)
    csv_filename = os.path.join(output_dir, f"{output_dir}-nuclei_normalized_summary.csv")
    df_output.to_csv(csv_filename, index=False)

    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUCCESSFUL")
    print(f"Master file saved to: {csv_filename}")
    print("=" * 60)
    print(df_output.to_string(index=False))


# --- Run Pipeline ---
# run_nuclei_normalized_pipeline("your_file.nd2")


# --- Execute Target Call ---
file_path = r"\\10.64.139.2\data\Temp - Please delete your files after copying\billy\image_data\2026-5-28 20260320 retake\phm topto light 96hrs.nd2"
run_nuclei_normalized_pipeline(file_path, output_dir= "optosos_light perk")

# --- Execute Target Call ---
file_path = r"\\10.64.139.2\data\Temp - Please delete your files after copying\billy\image_data\2026-5-28 20260320 retake\phm topto dark 96hrs.nd2"
run_nuclei_normalized_pipeline(file_path, output_dir= "optosos_dark perk" )

