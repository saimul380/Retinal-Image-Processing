from flask import Flask, render_template, request
from processing import process_image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    file.save(filepath)

    # Process the uploaded image
    results = process_image(filepath)

    return render_template(
    "index.html",
    image=filepath,
    clahe=results["clahe"],
    histogram=results["histogram"],
    green=results["green"],
    gaussian=results["gaussian"],
    median=results["median"],
    opening=results["opening"],
    closing=results["closing"],
    threshold=results["threshold"],
    vessel=results["vessel"],
    overlay=results["overlay"],
    processing_time=results["processing_time"],
    features=results["features"]
)


if __name__ == "__main__":
    app.run(debug=True)