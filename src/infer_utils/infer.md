# Inference and Evaluation for CTSM

This markdown file provides an overview of the inference process and evaluation metrics used to assess the performance of the Compression Temporal Shift Module (CTSM). The inference pipeline comprises multiple scripts for calculating various metrics such as PSNR, SSIM, LPIPS, FPS, Compression Rate, and VMAF.

---

## Inference Scripts and Their Functions

### 1. `inference.py`
This script is responsible for calculating the following metrics using the pre-trained CTSM model:
- **PSNR (Peak Signal-to-Noise Ratio):** Measures the reconstruction quality by comparing the original and reconstructed images.
- **SSIM (Structural Similarity Index Measure):** Assesses the perceived quality of reconstructed images based on structural information.
- **LPIPS (Learned Perceptual Image Patch Similarity):** Evaluates perceptual similarity using a pre-trained deep network.

These metrics are saved as `.npy` files in the specified output directory.

---

### 2. `generate_results.py`
This script generates:
- **Latent Space Representations:** Saves latent space visualizations for each frame.
- **Reconstructed Images:** Produces images reconstructed from latent representations.
- **Entropy Histograms:** Computes and plots entropy histograms for the latent space to assess information preservation.

The outputs include latent space images, reconstructed images, and entropy histograms saved in the specified results directory.

---

### 3. `compression_rate.py`
This script computes the **compression rate** of the CTSM model using:
- **Huffman Encoding:** Compresses latent representations to evaluate the efficiency of the model in reducing storage requirements.
- **Compression Factor (CF):** Calculated as the ratio of the original size to the compressed size.

The average compression factor is reported in the console and saved as JSON for further analysis.

---

### 4. `inference_fps.py`
This script calculates the **Frames Per Second (FPS)** during inference on a CPU:
- Measures the efficiency of the model in terms of processing speed.

---

### 5. `infer_video.py`
This script is used for:
- **Generating Videos:** Generated original and reconstructed videos using the pre-trained model.
- **Calculating VMAF (Video Multi-Method Assessment Fusion):** Assesses video quality using FFmpeg commands:

```bash
ffmpeg -i reconstructed.mp4 -i original.mp4 -lavfi libvmaf="log_path=vmaf_score.csv:log_fmt=csv" -f null -

```
