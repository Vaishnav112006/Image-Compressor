import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
from math import sqrt, log10
from encoder import compress_image_rgb, encode_coeffs_to_bin, compute_psnr, compute_ssim_safe

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff"}
MAX_CONTENT_LENGTH = 24 * 1024 * 1024  # 24 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "replace_with_a_random_secret_for_production"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            base, ext = os.path.splitext(filename)
            orig_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base}_orig{ext}")
            file.save(orig_path)

            pil = Image.open(orig_path).convert("RGB")
            img_rgb = np.array(pil)

            try:
                q = int(request.form.get("quality", 50))
            except ValueError:
                q = 50

            subsample = request.form.get("subsample", "off") == "on"

            rec_rgb, all_coeffs = compress_image_rgb(img_rgb, quality=q, subsample=subsample)

            comp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base}_Q{q}.png")
            cv2.imwrite(comp_path, cv2.cvtColor(rec_rgb, cv2.COLOR_RGB2BGR))

            p = compute_psnr(img_rgb, rec_rgb)
            s = compute_ssim_safe(img_rgb, rec_rgb)

            bin_name = f"{base}_Q{q}.bin"
            bin_path = os.path.join(app.config["UPLOAD_FOLDER"], bin_name)
            encode_coeffs_to_bin(all_coeffs, bin_path)

            orig_size = os.path.getsize(orig_path)
            comp_size = os.path.getsize(comp_path)
            bin_size = os.path.getsize(bin_path)
            reduction_enc = 100.0 * (orig_size - bin_size) / orig_size if orig_size>0 else 0.0
            reduction_img = 100.0 * (orig_size - comp_size) / orig_size if orig_size>0 else 0.0

            log = {
                "quality": q,
                "subsample": subsample,
                "orig_size": orig_size,
                "reconstructed_image_size": comp_size,
                "encoded_bin_size": bin_size,
                "psnr": round(p,2),
                "ssim": round(s,4),
                "reduction_encoded_percent": round(reduction_enc,2),
                "reduction_image_percent": round(reduction_img,2)
            }

            return render_template("index.html",
                                   original=os.path.basename(orig_path),
                                   compressed=os.path.basename(comp_path),
                                   binfile=os.path.basename(bin_path),
                                   quality=q,
                                   metrics=log)
    return render_template("index.html")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
