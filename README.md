# Image Denoising Pipeline

## Project Overview
A Colab-based research pipeline for removing synthetic noise from images.  
Implements and compares four denoising methods:
- **Convolutional Autoencoder**  
- **U-Net**  
- **Pix2Pix-style GAN**  
- **Swin Transformer Fine-tuner**  

Each model is trained on high-noise levels (Gaussian σ=50, Salt & Pepper density=0.15, Speckle variance=0.1), evaluated with PSNR/SSIM/MSE metrics, and visualized in Matplotlib. The best U-Net weights power a Gradio demo deployed on Hugging Face Spaces.

---
## Install required packages
```bash
pip install -r requirements.txt
```
Key dependencies (requirements.txt):

- torch & torchvision

- timm (Swin Transformer backbone)

- scikit-image (PSNR, SSIM, MSE)

- numpy, matplotlib, tqdm, Pillow

- gradio (demo UI)

- huggingface_hub (deployment)
---
## Usage Examples

### 1. Open and run the Colab notebook

- File: denoising_notebook.ipynb

- Execute cells in order:

  - Setup (paths, random seeds)

  - Noisy Data Generation (Gaussian, S&P, Speckle)

  - Dataset & DataLoader Definitions

  - Model & Training Function Definitions

  - Main Orchestration (ConvAE, U-Net)

  - GAN & SwinDenoiser Experiments

  - Save Best Weights

### 2. Inspect results

  - View training & test loss curves

- Examine PSNR / SSIM / MSE tables (pandas)

- Preview sample reconstructions (clean, noisy, denoised)

### 3. Run Gradio demo locally

```bash
python app.py
```
- Upload a noisy image (e.g., salt-pepper noise)

- View the denoised output in your browser
---
## Data Description

This project uses a custom “denoising” dataset derived from Kaggle’s PlantVillage:

- Clean images

  - Download via Kaggle Hub API:

```bash
import kagglehub
path = kagglehub.dataset_download("aastha2807/denoising")
print("Clean data path:", path)
```
   - Structure after extraction:

```bash
~/.cache/kagglehub/datasets/aastha2807/denoising/versions/1/
  ├── train/twenty_fivek_hr/
  └── test/twenty_fivek_hr/

```
- Noisy subsets

  - Generated in-notebook under:

```bash
/content/denoising_dataset_noisy_subset_single_g50/
  ├── train/
  │   ├── gaussian_50/
  │   ├── saltpepper_015/
  │   └── speckle_01/
  └── test/
      ├── gaussian_50/
      ├── saltpepper_015/
      └── speckle_01/
```
If you prefer manual setup:

- Download PlantVillage from Kaggle.

- Reorganize into the above train/test folders.

- Update CLEAN_DATA_BASE_PATH in Cell 1 of the notebook.
---
## Deploy

To deploy the Gradio app on Hugging Face Spaces:

- Ensure app.py and requirements.txt are up to date.

- Authenticate:
```bash
from huggingface_hub import login
login()
```

- Upload files:

```bash
from huggingface_hub import upload_file

hf_space_id = "your-username/denoising-space"

upload_file("app.py",            path_in_repo="app.py",            repo_id=hf_space_id, repo_type="space")
upload_file("requirements.txt",  path_in_repo="requirements.txt",  repo_id=hf_space_id, repo_type="space")
upload_file("unet_saltpepper_015_weights.pth",
            path_in_repo="unet_saltpepper_015_weights.pth",
            repo_id=hf_space_id, repo_type="space")

```

- Enable GPU in Space settings.

- Visit [Sprihttps://huggingface.co/spaces/your-username/denoising-spaceng](https://huggingface.co/spaces/your-username/denoising-space)

---
## Technologies

- Python – Core language

- PyTorch – Model building & training

- timm – Swin Transformer backbone

- scikit-image – Image quality metrics

- Matplotlib, tqdm – Visualization & progress bars

- Gradio – Web demo interface

- Hugging Face Hub – Model & app deployment

---

## Documentation

- Detailed explanations and code comments are in DIP_Project_7.ipynb.

- For the Gradio demo API, refer to docstrings in app.py.
## Dependencies & Installation

---
## Acknowledgments

- Kaggle PlantVillage dataset contributors

- PyTorch & Hugging Face open-source communities

- scikit-image developers for metrics

- Google Colab team for free GPU resources