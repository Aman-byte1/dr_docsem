"""Prompt templates for the DocSem pipeline."""

SYSTEM_PROMPT = """You are an expert at document-grounded quantitative reasoning. You will be given a document split into blocks, each starting with an identifier like b01:, and a user query. The query points to exactly one block that contains the target quantitative question together with the numbers needed to answer it.

Do the following:
1. Identify the block ID that contains the target question and its numbers.
2. Write a short Python program that computes the exact answer from the numbers in that block, and print the result.
3. Reply in EXACTLY this format and nothing else:

EVIDENCE: bXX
```python
# short code
print(result)
```
FINAL: <final answer as a plain number>
"""


def build_user_prompt(query: str, blocks: dict) -> str:
    """Render all document blocks plus the user query for the solver."""
    def key(bid: str) -> int:
        try:
            return int(bid[1:])
        except ValueError:
            return 999

    ordered = sorted(blocks.items(), key=lambda kv: key(kv[0]))
    block_lines = [f"{bid}: {text}" for bid, text in ordered]
    return (
        "DOCUMENT BLOCKS:\n"
        + "\n".join(block_lines)
        + f"\n\nUSER QUERY: {query}\n\n"
        + "Identify the evidence block, write the Python program, and give the final answer."
    )


VLM_OCR_PROMPT = (
    "Transcribe every text block on this document page. Each block begins with an "
    "identifier such as b01:. Output one line per block, formatted exactly as "
    "'bXX: <complete block text>'. Preserve all numbers, punctuation and wording "
    "exactly as they appear. Do not add any commentary."
)
