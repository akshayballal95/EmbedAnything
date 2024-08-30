import re
from typing import Dict, List
import uuid
from python.embed_anything import embed_anything
import time
from python.embed_anything import EmbedConfig, BertConfig, EmbedData
from python.embed_anything.vectordb import Adapter
from pinecone import Pinecone, ServerlessSpec
import os


class PineconeAdapter(Adapter):
    """
    Adapter class for interacting with Pinecone, a vector database service.
    """

    def __init__(self, api_key: str):
        """
        Initializes a new instance of the PineconeAdapter class.

        Args:
            api_key (str): The API key for accessing the Pinecone service.
        """
        super().__init__(api_key)
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = None

    def create_index(
        self,
        dimension: int,
        metric: str = "cosine",
        index_name: str = "anything",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    ):
        """
        Creates a new index in Pinecone.

        Args:
            dimension (int): The dimensionality of the embeddings.
            metric (str, optional): The distance metric to use for similarity search. Defaults to "cosine".
            index_name (str, optional): The name of the index. Defaults to "anything".
            spec (ServerlessSpec, optional): The serverless specification for the index. Defaults to AWS in us-east-1 region.
        """
        self.index_name = index_name
        self.pc.create_index(
            name=index_name, dimension=dimension, metric=metric, spec=spec
        )

    def delete_index(self, index_name: str):
        """
        Deletes an existing index from Pinecone.

        Args:
            index_name (str): The name of the index to delete.
        """
        self.pc.delete_index(name=index_name)

    def convert(self, embeddings: List[EmbedData]) -> List[Dict]:
        """
        Converts a list of embeddings into the required format for upserting into Pinecone.

        Args:
            embeddings (List[EmbedData]): The list of embeddings to convert.

        Returns:
            List[Dict]: The converted data in the required format for upserting into Pinecone.
        """
        data_emb = []

        for embedding in embeddings:
            data_emb.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding.embedding,
                    "metadata": {
                        "text": embedding.text,
                        "file": re.split(
                            r"/|\\", embedding.metadata.get("file_name", "")
                        )[-1],
                    },
                }
            )
        return data_emb

    def upsert(self, data: List[Dict]):
        """
        Upserts data into the specified index in Pinecone.

        Args:
            data (List[Dict]): The data to upsert into Pinecone.

        Raises:
            ValueError: If the index has not been created before upserting data.
        """

        data = self.convert(data)

        if not self.index_name:
            raise ValueError("Index must be created before upserting data")
        self.pc.Index(name=self.index_name).upsert(data)


start_time = time.time()

bert_config = BertConfig(
    model_id="sentence-transformers/all-MiniLM-L12-v2", chunk_size=100
)

# available model_ids:
# jinaai/jina-embeddings-v2-base-en,
# jinaai/jina-embeddings-v2-small-en,
# jinaai/jina-embeddings-v2-base-zh,
# jinaai/jina-embeddings-v2-small-de
# jina_config = JinaConfig(
#     model_id="jinaai/jina-embeddings-v2-small-en", revision="main", chunk_size=100
# )
embed_config = EmbedConfig(bert=bert_config)

data = embed_anything.embed_query(
    ["Hello world", "Hi"], embeder="Bert", config=embed_config
)
# data = embed_anything.embed_directory(
#     "test_files", embeder="Bert", extensions=["pdf"], config=embed_config
# )


api_key = os.environ.get("PINECONE_API_KEY")
index_name = "anything"
pinecone_adapter = PineconeAdapter(api_key)

try:
    pinecone_adapter.delete_index("anything")
except:
    pass

pinecone_adapter.create_index(dimension=384, metric="cosine")


data_file = embed_anything.embed_file(
    "./test_files/test.pdf",
    embeder="Bert",
    config=embed_config,
    adapter=pinecone_adapter,
)

print(data)
print(pinecone_adapter.pc.Index(name="anything").list_paginated())
end_time = time.time()
print("Time taken: ", end_time - start_time)