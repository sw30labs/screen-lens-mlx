"""
CLIP Embedding Module.

Generates visual and text embeddings using OpenCLIP for semantic search.
Supports both image embedding (for frames) and text embedding (for queries).
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import EmbeddingConfig


class _PublicHubAuthWarningFilter(logging.Filter):
    """Hide Hugging Face's advisory warning for public OpenCLIP weights."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests to the HF Hub" not in record.getMessage()


class CLIPEmbedder:
    """Manages CLIP model loading and embedding generation."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the CLIP model."""
        if self._model is not None:
            return

        import torch
        import open_clip
        from PIL import Image as PILImage  # noqa: ensure available

        device = self.config.device
        # Validate device
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
            print("MPS not available, falling back to CPU")
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA not available, falling back to CPU")

        self.config.device = device

        print(f"Loading OpenCLIP model {self.config.model_name} on {device}...")
        # The configured checkpoint is public. Newer Hub servers emit an
        # advisory warning (often twice under TUI log forwarding) without a
        # token even though the download is valid. Actual download, auth, and
        # rate-limit failures still propagate normally.
        hub_logger = logging.getLogger("huggingface_hub.utils._http")
        auth_filter = _PublicHubAuthWarningFilter()
        hub_logger.addFilter(auth_filter)
        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.config.model_name,
                pretrained=self.config.pretrained,
                device=device,
            )
        finally:
            hub_logger.removeFilter(auth_filter)
        self._tokenizer = open_clip.get_tokenizer(self.config.model_name)
        self._model.eval()
        print("CLIP model loaded.")

    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """
        Generate CLIP embeddings for a list of images.

        Returns: numpy array of shape (N, embedding_dim)
        """
        import torch
        from PIL import Image

        self._load_model()
        device = self.config.device
        all_embeddings = []

        for i in tqdm(
            range(0, len(image_paths), self.config.batch_size),
            desc="Generating image embeddings",
        ):
            batch_paths = image_paths[i : i + self.config.batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(self._preprocess(img))
                except Exception as e:
                    print(f"Warning: Failed to load {p}: {e}")
                    # Use a blank image as placeholder
                    images.append(self._preprocess(Image.new("RGB", (224, 224))))

            batch_tensor = torch.stack(images).to(device)
            with torch.inference_mode():
                features = self._model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def embed_text(self, queries: list[str]) -> np.ndarray:
        """
        Generate CLIP embeddings for text queries.

        Returns: numpy array of shape (N, embedding_dim)
        """
        import torch

        self._load_model()
        device = self.config.device

        tokens = self._tokenizer(queries).to(device)
        with torch.inference_mode():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        self._load_model()
        # ViT-B-32 = 512, ViT-L-14 = 768
        return self._model.visual.output_dim
