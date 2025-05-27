# CS2-SGNN-VTAC

## Note on File Size Limitations

Due to a 50MB maximum file upload limit, we had to exclude the `.csv` and `.npz` files from this repository. These files take up too much space even when compressed. You can download them from the following external links:

- **Link to Elliptic++ data**: [Google Drive](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=drive_link)  
- **Link to NPZ files**: [Google Drive](https://drive.google.com/drive/folders/1hM75coqXuSORUVZXhs18NKuEkqCml5t9?usp=sharing)

Please download these files and place them in the appropriate locations before running the scripts.

---

## Running VTAC Scripts

To run VTAC scripts, set up a virtual python env with the following version Python 3.10.8, then install requirements.txt and use the following commands:

1. **Set root folder path**:
   ```powershell
   $env:PYTHONPATH = "<path_to_folder>\CS2-SGNN-VTAC\"
   python ./VTAC/Our_VTAC.py
