from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()

# Edit this to add the chain you want to add
from hybrid_search_weaviate import chain as hybrid_search_weaviate_chain
add_routes(app, hybrid_search_weaviate_chain, path="/hybrid-search-weaviate")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
