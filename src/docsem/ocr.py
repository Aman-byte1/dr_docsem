"""OCR engines: RapidOCR (light, default) and optional VLM transcription."""

import numpy as np

from .blocks import parse_blocks
from .prompts import VLM_OCR_PROMPT

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        try:
            # Cap intra-op threads: without this, every worker process spawns
            # an ONNX thread-pool sized to all cores (e.g. 8 workers x 96
            # threads = 768 threads on 96 cores -> oversubscription stall).
            _OCR = RapidOCR(intra_op_num_threads=4)
        except TypeError:
            _OCR = RapidOCR()
    return _OCR


def ocr_page_lines(img: np.ndarray) -> list:
    """OCR one page image into reading-order text lines."""
    ocr = get_ocr()
    result, _ = ocr(img)
    if not result:
        return []
    items = []
    for box, text, _score in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), min(xs), str(text)))
    items.sort(key=lambda t: (t[0], t[1]))

    line_gap = max(12.0, 0.010 * img.shape[0])
    lines, cur, last_y = [], [], None
    for y, x, text in items:
        if last_y is None or (y - last_y) <= line_gap:
            cur.append((x, text))
            last_y = y if last_y is None else min(last_y, y)
        else:
            cur.sort()
            lines.append(" ".join(t for _, t in cur))
            cur = [(x, text)]
            last_y = y
    if cur:
        cur.sort()
        lines.append(" ".join(t for _, t in cur))
    return lines


def ocr_document(pdf_path, dpi: int = 220) -> str:
    from .pdf_images import page_images

    pages = []
    for img in page_images(pdf_path, dpi=dpi):
        pages.append("\n".join(ocr_page_lines(img)))
    return "\n".join(pages)


def ocr_document_blocks(pdf_path, dpi: int = 220) -> dict:
    return parse_blocks(ocr_document(pdf_path, dpi=dpi))


def text_layer_document_blocks(pdf_path) -> dict:
    from .pdf_images import page_texts

    return parse_blocks("\n".join(page_texts(pdf_path)))


def vlm_document_blocks(pdf_path, model_id: str, dpi: int = 220, max_new_tokens: int = 1024) -> dict:
    """Transcribe pages with a Qwen2.5-VL model (use as the retry path for hard docs)."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from .pdf_images import page_images

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    pages = []
    for img in page_images(pdf_path, dpi=dpi):
        image = Image.fromarray(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": VLM_OCR_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        pages.append(processor.batch_decode(trimmed, skip_special_tokens=True)[0])
    return parse_blocks("\n".join(pages))
