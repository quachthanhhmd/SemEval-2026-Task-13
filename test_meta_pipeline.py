import torch
import logging
import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import GraphCodeBERTDomainModel
from src.dataset import CodeDataset, DomainRegistry
from src.features import AgnosticFeatureExtractor
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Test Model
    logger.info("Testing Model Forward Pass...")
    model = GraphCodeBERTDomainModel(
        num_generators=5,
        num_languages=10,
        num_style=11,
        model_name="microsoft/graphcodebert-base"
    ).to(device)
    
    batch_size = 4
    max_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, max_len)).to(device)
    attention_mask = torch.ones((batch_size, max_len)).to(device)
    extra_features = torch.randn((batch_size, 11)).to(device)
    
    out = model(input_ids, attention_mask, extra_features)
    
    logger.info(f"Model keys: {out.keys()}")
    assert "label_logits" in out
    assert "generator_logits" in out
    assert "language_logits" in out
    assert "projection" in out
    assert out["label_logits"].shape == (batch_size, 2)
    assert out["generator_logits"].shape == (batch_size, 5)
    assert out["language_logits"].shape == (batch_size, 10)
    assert out["projection"].shape == (batch_size, 128)
    logger.info("Model Forward Pass Success!")

    # 2. Test Dataset (Minimal)
    logger.info("Testing Dataset Logic...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    registry = DomainRegistry()
    registry.language2id = {"python": 0, "java": 1}
    registry.generator2id = {"human": 0, "gpt": 1}
    registry.domain2id = {("python", "human"): 0, ("java", "gpt"): 1}
    
    # We won't run full extractor here to avoid loading the big model
    extractor = None 
    
    data = [
        {"code": "def hello(): print('hi')", "label": 0, "language": "python", "generator": "human"},
        {"code": "public class Main { }", "label": 1, "language": "java", "generator": "gpt"}
    ]
    
    ds = CodeDataset(data, tokenizer, registry, extractor=extractor, max_length=128)
    sample = ds[0]
    
    logger.info(f"Dataset sample keys: {sample.keys()}")
    assert "input_ids" in sample
    assert "extra_features" in sample
    assert sample["extra_features"].shape == (11,)
    logger.info("Dataset Logic Success!")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
