import asyncio
import os
import re
from dataclasses import dataclass

import numpy as np
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from .._utils import logger
from ..base import BaseVectorStorage


def _sanitize_collection_name(name: str) -> str:
    """Milvus collection names: alphanumeric and underscores only, must start with letter/underscore."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = f"_{name}"
    return name[:255] or "default"


@dataclass
class MilvusVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._max_batch_size = self.global_config.get(
            "embedding_batch_num",
            self.global_config.get("llm", {}).get("embedding_batch_num", 32),
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

        prefix = _sanitize_collection_name(
            os.path.basename(self.global_config["working_dir"])
        )
        self._collection_name = f"{prefix}_{self.namespace}"

        uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
        db_name = os.environ.get("MILVUS_DB_NAME", "default")
        self._client = MilvusClient(uri=uri, db_name=db_name)

        if not self._client.has_collection(self._collection_name):
            dim = self.embedding_func.embedding_dim
            schema = CollectionSchema(
                fields=[
                    FieldSchema("id", DataType.VARCHAR, max_length=512, is_primary=True),
                    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
                ],
                enable_dynamic_field=True,
            )
            self._client.create_collection(
                self._collection_name, schema=schema
            )
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                "vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 256},
            )
            self._client.create_index(self._collection_name, index_params)
            logger.info(
                f"Created Milvus collection {self._collection_name} (dim={dim})"
            )
        else:
            logger.info(f"Using existing Milvus collection {self._collection_name}")

        self._client.load_collection(self._collection_name)

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not data:
            logger.warning("You insert an empty data to vector DB")
            return []

        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings = np.concatenate(
            await asyncio.gather(*[self.embedding_func(batch) for batch in batches])
        )

        rows = []
        for i, d in enumerate(list_data):
            row = {**d, "vector": embeddings[i].tolist()}
            rows.append(row)

        self._client.upsert(self._collection_name, rows)
        return [d["id"] for d in list_data]

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = await self.embedding_func([query])
        embedding = embedding[0]

        results = self._client.search(
            self._collection_name,
            data=[embedding.tolist()],
            limit=top_k,
            output_fields=["*"],
            search_params={"metric_type": "COSINE", "params": {"ef": max(top_k, 50)}},
        )

        if not results or not results[0]:
            return []

        output = []
        for hit in results[0]:
            similarity = hit["distance"]  # COSINE metric returns similarity (0-1)
            if similarity < self.cosine_better_than_threshold:
                continue
            entity = hit.get("entity", {})
            doc_id = entity.pop("id", hit["id"])
            output.append(
                {
                    **entity,
                    "id": doc_id,
                    "__id__": doc_id,
                    "distance": similarity,
                    "__metrics__": similarity,
                }
            )

        return output

    async def index_done_callback(self):
        self._client.flush(self._collection_name)


@dataclass
class MilvusVectorDBVideoSegmentStorage(MilvusVectorDBStorage):
    pass
