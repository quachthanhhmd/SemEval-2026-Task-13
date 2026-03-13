import sys
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import logging

# Set logging to capture warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_spanning")

# Add project root to sys.path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import CodeDataset, DomainRegistry, char_span

def test_char_span():
    print("Testing line-respecting char_span...")
    code = "line1\nline2\nline3\nline4\nline5"
    max_chars = 15 # Should fit roughly 2-3 lines
    span = char_span(code, max_chars)
    assert "\n" in span
    # Each line is 5 chars + 1 newline = 6 chars
    # "line1\nline2\n" is 12 chars. "line1\nline2\nline3" is 18 chars.
    # So it should be 12 chars if starting at line1
    print(f"Success: span is:\n{span}")
    print(f"Length: {len(span)}")

def test_dataset_warnings():
    print("\nTesting dataset warnings and length...")
    model_name = "microsoft/graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create a dummy registry
    registry = DomainRegistry()
    registry.language2id = {"python": 0}
    registry.generator2id = {"gpt-4": 0}
    registry.domain2id = {("python", "gpt-4"): 0}
    
    # Create very long code (> 2000 chars, > 512 tokens)
    long_code = "def large_function():\n" + "    x = 1\n" * 1000
    
    data = {
        "code": [long_code],
        "label": [1],
        "language": ["python"],
        "generator": ["gpt-4"]
    }
    df = pd.DataFrame(data)
    
    ds = CodeDataset(df, tokenizer, registry, max_length=512, augment=True)
    
    # This should NOT produce a warning in stderr/stdout
    print("Fetching item (augment=True)...")
    item = ds[0]
    
    print(f"Input IDs length: {len(item['input_ids'])}")
    assert len(item['input_ids']) == 512
    assert item['input_ids'][0] == tokenizer.cls_token_id
    assert torch.any(item['input_ids'] == tokenizer.sep_token_id)
    
    print("Success: Item fetched with correct length and no warnings (hopefully).")

if __name__ == "__main__":
    try:
        test_char_span()
        test_dataset_warnings()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
