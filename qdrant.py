from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import TfidfVectorizer

import uuid
import os
import numpy as np

import joblib
from pathlib import Path

from typing import List

from log_config import logger

class VectorStore:
    """
        未来的工作：
            为了加速检索速度，减少token使用，需要限制输入文本的大小。
            同时，我们计划降低embedding vector维数，精简 payload 大小，并设置 on_disk_payload = False，
            将payload加载到内存中处理。

        数据库使用前必须先load！
    """
    def __init__(self, embedding_model, collection_name="innovations_knowledge"):
        

        # For dense vector.
        self.embedding_model = embedding_model

        self.dense_dimensions = self.embedding_model.dimensions

        self.collection_name = collection_name

        url= os.getenv("DOCKER_QDRANT_URL", "http://localhost:6333")
        self.qdrant = QdrantClient(url = url)

        self.init_collection()
        self.init_sparse_vectorizer()

    def init_sparse_vectorizer(self):
        # For sparse vector.
        self.sparse_dimensions = 1000

        # Re-fit TfidfVectorizer every `sparse_update_limit` inputs.
        self.sparse_update_counter = 0
        self.sparse_update_rm_counter = 0
        self.sparse_update_limit = 300

        self.sparse_vectorizer_path = Path(__file__).resolve().parent \
                                        / "qdrant_data"  / "sparse_vectorizer.joblib"
        
        if self.sparse_vectorizer_path.exists():
            self.sparse_vectorizer = joblib.load(self.sparse_vectorizer_path)
            logger.info("Loaded existed sparse vectorizer !")
        else:
            self.sparse_vectorizer = TfidfVectorizer(
                max_features = self.sparse_dimensions,
                max_df = 0.8,
                min_df = 1,
                stop_words = "english",
                ngram_range=(1, 2),
                norm = "l2"
            )
            logger.warning("Vectorizer inited. Must fit it before using it.")


    def init_collection(self):
        
        # Initialize collection if not exists.
        if not self.qdrant.collection_exists(collection_name = self.collection_name):

            self.qdrant.create_collection(
                collection_name = self.collection_name,
                vectors_config = {
                    "dense": models.VectorParams(size = self.dense_dimensions, \
                                                     distance = models.Distance.COSINE)
                },
                sparse_vectors_config = {
                    "sparse": models.SparseVectorParams(
                        index = models.SparseIndexParams(on_disk = False),
                        modifier = models.Modifier.IDF
                    )
                },

                on_disk_payload = False,

                hnsw_config=models.HnswConfigDiff(  
                    m = 16,             
                    ef_construct = 100  
                )
            )
        
            self.qdrant.create_payload_index(
                collection_name = self.collection_name,
                field_name = "text",
                field_schema = models.PayloadSchemaType.TEXT
            )
            logger.info("Initialized Qdrant Collection !")
        else:
            logger.info("Loaded existed Qdrant Collection !")


    @staticmethod
    def generate_hash_id(text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    def get_sparse_vectors(self, texts:List[str]) -> List[dict]:

        matrix = self.sparse_vectorizer.transform(texts)

        return [{"indices": row.indices.tolist(), "values":row.data.tolist()} for row in matrix]
    
    def refit_sparse_vectorizer(self):
        batch_size = 300
        offset = None
        ids = []
        texts = []

        while True:
            points, offset = self.qdrant.scroll(
                collection_name = self.collection_name,
                limit = batch_size,
                with_payload = ["text"],
                offset = offset
            )

            if not points:
                break
            
            for point in points:
                ids.append(point.id)
                texts.append(point.payload.get("text"))

        self.sparse_vectorizer.fit(texts)

        joblib.dump(self.sparse_vectorizer, self.sparse_vectorizer_path)
        logger.info("Sparse Vectorizer saved!")

        sparse_vectors = self.get_sparse_vectors(texts)

        points = []
        for id, sparse_vec in zip(ids, sparse_vectors):
            points.append(models.PointVectors(
                id = id,
                vector = {"sparse": models.SparseVector(\
                    indices = sparse_vec["indices"], values = sparse_vec["values"])}
            ))
        
        self.qdrant.update_vectors(
            collection_name = self.collection_name,
            points = points
        )

        self.sparse_update_counter = 0
        self.sparse_update_rm_counter = 0

        logger.info("Sparse vectors updated.")



    def load(self, texts: List[str], dense_vectors:List[List[float]]):

        if not texts:
            logger.error("[VectorStore.load] Must provide texts for initializing !")

        self.sparse_vectorizer.fit(texts)
        joblib.dump(self.sparse_vectorizer, self.sparse_vectorizer_path)
        logger.info("Sparse Vectorizer saved!")

        self.qdrant.delete_collection(collection_name = self.collection_name)
        self.init_collection()

        sparse_vectors = self.get_sparse_vectors(texts)

        points = []
        for text, dense_vec, sparse_vec in zip(texts, dense_vectors, sparse_vectors):
            id = self.generate_hash_id(text)

            point = models.PointStruct(
                id = id,
                vector = {
                    "sparse": models.SparseVector(indices = sparse_vec["indices"], values = sparse_vec["values"]),
                    "dense": dense_vec
                },
                payload = {"text": text}
            )

            points.append(point)

        self.qdrant.upload_points(
            collection_name = self.collection_name,
            points = points,
            parallel = 4,
            max_retries = 3
        )

    
    def remove(self, texts: List[str]):

        ids = [self.generate_hash_id(text) for text in texts]
                    
        # Idempotent operation, automatically skips when id doesn't exist.

        try:
            self.qdrant.delete(
                collection_name = self.collection_name,
                points_selector = models.PointIdsList(points = ids)
            )

            self.sparse_update_rm_counter += len(ids)
            
            logger.info(f"[Qdrant.delete] try to delete {len(ids)} point(s)")

            if(self.sparse_update_rm_counter > self.sparse_update_limit):
                self.refit_sparse_vectorizer()

        except Exception as  e:
            logger.warning(f"[Qdrant.delete] Something may went wrong:{e}")


    def embed_and_save(self, texts: List[str]):

        # Avoid duplicate inputs.
        accepted_text_count = 0
        
        points = []
        for text in texts:
            id = self.generate_hash_id(text)

            # Check for existence by ID (lightweight)
            try:
                retrieved_points = self.qdrant.retrieve(
                    collection_name = self.collection_name,
                    ids = [id],
                    with_vectors = False,
                    with_payload = False
                )
                if retrieved_points:
                    continue
            except Exception as e:
                logger.warning(f"[Qdrant.retrieve] Something may went wrong:{e}") 

            sparse_vec = self.get_sparse_vectors([text])[0]
            dense_vec = self.embedding_model.embed_query(text)

            point = models.PointStruct(
                id = id,
                vector = {
                    "sparse": sparse_vec,
                    "dense": dense_vec
                },
                payload = {"text": text}
            )

            points.append(point)

            accepted_text_count += 1

        self.qdrant.upload_points(
            collection_name = self.collection_name,
            points = points,
            parallel = 4,
            max_retries = 3
        )

        logger.info(f"Inserted {accepted_text_count} new vectors to Qdrant, {len(texts) - accepted_text_count} duplicate texts skipped.")
        
        self.sparse_update_counter += accepted_text_count

        if(self.sparse_update_counter > self.sparse_update_limit):
            self.refit_sparse_vectorizer()
        

    # Single query. Hybrid search.
    def search(self, query: str, limit = 30, min_result_count = 5):
        
        dense_vec = self.embedding_model.embed_query(query)
        sparse_vec = self.get_sparse_vectors([query])[0]

        # Reciprocal Rank Fusion (RRF)

        results = self.qdrant.query_points(
            collection_name = self.collection_name,

            prefetch = [
                models.Prefetch(
                    query = models.SparseVector(\
                                indices = sparse_vec["indices"], values = sparse_vec["values"]),
                    using = "sparse",
                    limit = limit
                ),
                models.Prefetch(
                    query = dense_vec,
                    using = "dense",
                    limit = limit
                )
            ],

            query = models.FusionQuery(fusion = models.Fusion.RRF),

            with_payload=["text"],
            with_vectors = False
        ).points

        if not results:
            logger.warning(f"[Qdrant search] There isn't any related result for {query}")
            return [],[]

        result_texts = []
        result_scores = []

        for result in results:
            result_texts.append(result.payload["text"])
            result_scores.append(result.score)

        return result_texts, result_scores
    