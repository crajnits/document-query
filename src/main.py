import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys
import pathlib
import pydantic

import transformers
import transformers_pipeline
import document
import ocr_reader

@pydantic.validate_call
def get_logger(prefix: str):
    log = logging.getLogger(prefix)
    log.propagate = False
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log

log = get_logger("query_system")

def query_generator(args):
    paths = []
    if pathlib.Path(args.path).is_dir():
        for root, dirs, files in os.walk(args.path):
            for fname in files:
                if (pathlib.Path(root) / fname).is_dir():
                    continue
                paths.append(pathlib.Path(root) / fname)
    else:
        paths.append(args.path)

    docs = []
    for p in paths:
        try:
            log.info(f"Loading {p}")
            docs.append((p, document.load_document(str(p), ocr_reader=args.ocr, use_embedded_text=args.use_embedded_text)))
        except document.UnsupportedDocument as e:
            log.warning(f"Cannot load {p}: {e}. Skipping...")

    log.info(f"Done loading {len(docs)} file(s).")
    if not docs:
        return

    log.info("Loading pipelines.")

    nlp = transformers_pipeline.pipeline("document-queries", model=args.checkpoint)
    log.info("Ready to start evaluating!")

    max_fname_len = max(len(str(p)) for (p, _) in docs)
    max_question_len = max(len(q) for q in args.questions) if len(args.questions) > 0 else 0
    for i, (p, d) in enumerate(docs):
        if i > 0 and len(args.questions) > 1:
            print("")

        for q in args.questions:
            try:
                response = nlp(question=q, **d.context)
                if isinstance(response, list):
                    response = response[0] if len(response) > 0 else None
            except Exception:
                log.error(f"Failed while processing {str(p)} on question: '{q}'")
                raise

            answer = response["answer"] if response is not None else "NULL"
            print(f"{str(p):<{max_fname_len}} {q:<{max_question_len}}: {answer}")

def main():
    """The main routine."""

    parser = argparse.ArgumentParser(description="document question answering system")

    parser.add_argument(
        "questions", default=[], nargs="*", type=str, help="One or more questions to ask of the documents"
    )

    parser.add_argument("path", type=str, help="The file or directory to scan")

    parser.add_argument("--verbose", "-v", default=False, action="store_true")

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"A custom model checkpoint to use (other than {transformers_pipeline.PIPELINE_DEFAULTS['document-queries']})",
    )

    parser.add_argument(
        "--ocr", choices=list(ocr_reader.OCR_MAPPING.keys()), default=None, help="The OCR engine you would like to use"
    )
    parser.add_argument(
        "--ignore-embedded-text",
        dest="use_embedded_text",
        action="store_false",
        help="Do not try and extract embedded text from document types that might provide it (e.g. PDFs)",
    )

    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    if not args.verbose:
        transformers.logging.set_verbosity_error()
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=level)

    return query_generator(args)

if __name__ == "__main__":
    sys.exit(main())
