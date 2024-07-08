# Document Query System

This implementation leverages the Hugging Face Transformers library, streamlining the code for
transformer-based models. It currently utilizes the pre-trained "layoutlm-document-qa" pipeline
model for question answering tasks on document layouts. This model can be further fine-tuned for
enhanced performance. Additionally, "pix2struct-docvqa-large" is another option for structured
queries.

The model takes both image and text as input, where the image provides visual context and the
text helps correlate with the questions. Currently, support for PDFs and images is included,
with easy extensibility for other document types like text files, spreadsheets, and custom
document classes.

- The first time you run this script, it will download the model checkpoints from Hugging Face,
so it may take longer to execute. To reduce the project size, the checkpoint file is not included.

## Getting Started

  - Code is tested with Python 3.10

  - Setup python virtual env and activate it.

  - Install python packages.
    ```
    python -m pip install -r requirements.txt
    ```

  - Install Tesseract-OCR and set PATH. (https://github.com/tesseract-ocr/tesseract)

  - Download, unzip Poppler and set PATH. (Only requierd for PDF file)

  - Run Examples
    ```
    python src/main.py "What is the invoice number?" data/invoice.png
    python src/main.py "Who authored this paper?" data/2210.03347v2.pdf
    python src/main.py "What is the purchase amount?" data/purchase.jpg
    ```
