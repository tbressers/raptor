import logging
import openai
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from dotenv import load_dotenv
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, engine="text-embedding-ada-002"):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.engine = engine

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], engine=self.engine)["data"][0]["embedding"]

class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)