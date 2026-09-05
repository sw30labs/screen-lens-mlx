"""
Vector Store Module.

Manages ChromaDB for storing and searching frame embeddings + captions.
Supports both CLIP visual embeddings and text-based caption search.
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.config import VectorDBConfig


class ScreenLensVectorStore:
    """ChromaDB-backed vector store for video frame search."""

    def __init__(self, config: Optional[VectorDBConfig] = None):
        self.config = config or VectorDBConfig()
        persist_dir = str(Path(self.config.persist_directory).resolve())
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = None

    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
        return self._collection

    def add_frames(
        self,
        frames_meta: list[dict],
        embeddings: np.ndarray,
    ):
        """
        Add frame data to the vector store.

        Args:
            frames_meta: List of frame metadata dicts (must have frame_id, timestamp, path, caption)
            embeddings: numpy array of CLIP embeddings, shape (N, dim)
        """
        collection = self._get_collection()

        ids = [f"frame_{f['frame_id']:06d}" for f in frames_meta]

        # Store rich metadata for retrieval
        metadatas = []
        for f in frames_meta:
            metadatas.append({
                "frame_id": f["frame_id"],
                "timestamp": f["timestamp"],
                "timestamp_str": f.get("timestamp_str", ""),
                "path": f["path"],
                "width": f.get("width", 0),
                "height": f.get("height", 0),
                "caption": f.get("caption", ""),
            })

        # Use captions as documents for text-based search
        documents = [f.get("caption", "") for f in frames_meta]

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents,
        )

        print(f"Added {len(ids)} frames to vector store '{self.config.collection_name}'")

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search for similar frames using a CLIP embedding vector.

        Returns list of results with metadata and distance scores.
        """
        collection = self._get_collection()

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        return self._format_results(results)

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search for frames using text query against captions.

        Uses ChromaDB's built-in text search on the documents (captions).
        """
        collection = self._get_collection()

        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        return self._format_results(results)

    def _format_results(self, raw_results: dict) -> list[dict]:
        """Format ChromaDB results into clean dicts."""
        formatted = []
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return formatted

        for i, id_ in enumerate(raw_results["ids"][0]):
            meta = raw_results["metadatas"][0][i] if raw_results["metadatas"] else {}
            doc = raw_results["documents"][0][i] if raw_results["documents"] else ""
            dist = raw_results["distances"][0][i] if raw_results["distances"] else 0

            formatted.append({
                "id": id_,
                "score": 1 - dist if dist <= 1 else 1 / (1 + dist),  # Convert distance to similarity
                "distance": dist,
                "caption": doc,
                **meta,
            })

        return formatted

    def get_all_frames(self) -> list[dict]:
        """Retrieve all frames from the collection."""
        collection = self._get_collection()
        results = collection.get(include=["metadatas", "documents"])

        frames = []
        for i, id_ in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            doc = results["documents"][i] if results["documents"] else ""
            frames.append({"id": id_, "caption": doc, **meta})

        return sorted(frames, key=lambda x: x.get("timestamp", 0))

    def count(self) -> int:
        """Return the number of frames in the store."""
        return self._get_collection().count()

    def reset(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(self.config.collection_name)
        self._collection = None
        print(f"Reset collection '{self.config.collection_name}'")
