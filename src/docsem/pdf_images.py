"""PDF page rendering and text-layer detection."""

import numpy as np
import pymupdf


def has_text_layer(pdf_path) -> bool:
    doc = pymupdf.open(str(pdf_path))
    try:
        for page in doc:
            if page.get_text().strip():
                return True
        return False
    finally:
        doc.close()


def page_texts(pdf_path):
    doc = pymupdf.open(str(pdf_path))
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def page_images(pdf_path, dpi: int = 220):
    """Yield RGB numpy arrays, one per page."""
    doc = pymupdf.open(str(pdf_path))
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = img[:, :, :3]
            yield img
    finally:
        doc.close()
