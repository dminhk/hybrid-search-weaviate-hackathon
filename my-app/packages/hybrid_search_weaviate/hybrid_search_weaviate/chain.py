import os

import weaviate
from langchain.chat_models import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.pydantic_v1 import BaseModel

# Check env vars
if os.environ.get("WEAVIATE_API_KEY", None) is None:
    raise Exception("Missing `WEAVIATE_API_KEY` environment variable.")

if os.environ.get("WEAVIATE_URL", None) is None:
    raise Exception("Missing `WEAVIATE_URL` environment variable.")

if os.environ.get("COHERE_API_KEY", None) is None:
    raise Exception("Missing `COHERE_API_KEY` environment variable.")

# Initialize the retriever
WEAVIATE_INDEX_NAME = os.environ.get("WEAVIATE_INDEX", "LangChain")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_client_secret,
    additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY
    }
)

client.schema.delete_all()

class_obj = {
    "class": "LangChain",
    "vectorizer": "text2vec-cohere",
    "moduleConfig": {
        "reranker-cohere": {
            "model": "rerank-multilingual-v2.0",
        },
    }
}
client.schema.create_class(class_obj)

retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name=WEAVIATE_INDEX_NAME,
    text_key="text",
    k=10,
    alpha=0.50,
    attributes=[],
    create_schema_if_missing=True,
)

# Ingest code - you may need to run this the first time
# Web scraping & QA use case
# Load

from langchain.document_loaders import AsyncHtmlLoader

# Weaviate 2023 blogs
urls = [
    "https://weaviate.io/blog/moonsift-story",
    "https://weaviate.io/blog/weaviate-1-22-release",
    "https://weaviate.io/blog/hacktoberfest-2023",
    "https://weaviate.io/blog/collections-python-client-preview",
    "https://weaviate.io/blog/confluent-and-weaviate",
    "https://weaviate.io/blog/pq-rescoring",
    "https://weaviate.io/blog/weaviate-gorilla-part-1",
    "https://weaviate.io/blog/hybrid-search-fusion-algorithms",
    "https://weaviate.io/blog/weaviate-1-21-release",
    "https://weaviate.io/blog/distance-metrics-in-vector-search",
    "https://weaviate.io/blog/what-is-a-vector-database",
    "https://weaviate.io/blog/healthsearch-demo",
    "https://weaviate.io/blog/automated-testing",
    "https://weaviate.io/blog/weaviate-1-20-release",
    "https://weaviate.io/blog/multimodal-models",
    "https://weaviate.io/blog/llamaindex-and-weaviate",
    "https://weaviate.io/blog/multi-tenancy-vector-search",
    "https://weaviate.io/blog/llms-and-search",
    "https://weaviate.io/blog/embedded-local-weaviate",
    "https://weaviate.io/blog/private-llm",
    "https://weaviate.io/blog/ingesting-pdfs-into-weaviate",
    "https://weaviate.io/blog/announcing-palm-modules",
    "https://weaviate.io/blog/generative-feedback-loops-with-llms",
    "https://weaviate.io/blog/weaviate-1-19-release",
    "https://weaviate.io/blog/wcs-public-beta",
    "https://weaviate.io/blog/how-to-chatgpt-plugin",
    "https://weaviate.io/blog/authentication-in-weaviate",
    "https://weaviate.io/blog/autogpt-and-weaviate",
    "https://weaviate.io/blog/ranking-models-for-better-search",
    "https://weaviate.io/blog/weaviate-retrieval-plugin",
    "https://weaviate.io/blog/monitoring-weaviate-in-production"
    "https://weaviate.io/blog/what-are-llms",
    "https://weaviate.io/blog/ann-algorithms-tiles-enocoder",
    "https://weaviate.io/blog/ann-algorithms-hnsw-pq",
    "https://weaviate.io/blog/weaviate-1-18-release",
    "https://weaviate.io/blog/solution-to-tl-drs",
    "https://weaviate.io/blog/combining-langchain-and-weaviate",
    "https://weaviate.io/blog/what-to-expect-from-weaviate-in-2023",
    "https://weaviate.io/blog/weaviate-podcast-search",
    "https://weaviate.io/blog/how-ai-creates-art",
    "https://weaviate.io/blog/vector-embeddings-explained",
    "https://weaviate.io/blog/pulling-back-the-curtains-on-text2vec",
    "https://weaviate.io/blog/hybrid-search-explained",
]

loader = AsyncHtmlLoader(urls)
docs = loader.load()

from langchain.document_transformers import Html2TextTransformer
html2text = Html2TextTransformer()
data = html2text.transform_documents(docs)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
retriever.add_documents(all_splits)


# RAG prompt
template = """
You are a Weaviate Vector Database expert.
Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatCohere(cohere_api_key=os.getenv('COHERE_API_KEY'))
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
