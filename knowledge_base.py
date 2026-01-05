import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from pathlib import Path
from huggingface_hub import snapshot_download
from get_models import ensure_model_dir

MODELS = {
    "BAAI/bge-reranker-base": Path("models/BAAI_bge_reranker"),
    "sentence-transformers/all-mpnet-base-v2": Path("models/all-mpnet-base-v2"),
}

def load_documents(directory_path: str) -> list:
    docs = []

    # Define which headers to capture (## and ### for your case)
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ]

    # Loop through files
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            for page in pages:
                text = page.page_content

                # Use MarkdownHeaderTextSplitter
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                split_docs = splitter.split_text(text)

                # Add metadata
                for d in split_docs:
                    d.metadata.update({
                        "source": f"{filename} (page {page.metadata.get('page', 'unknown')})"
                    })

                docs.extend(split_docs)

    return docs

def setup_retriever(docs: list) -> ContextualCompressionRetriever:
    # change force=True if you want to force re-download
    mpnet_local, _ = ensure_model_dir("sentence-transformers/all-mpnet-base-v2", Path("models/all-mpnet-base-v2"))
    reranker_local, _ = ensure_model_dir("BAAI/bge-reranker-base", Path("models/BAAI_bge_reranker"))

    # pass those local paths into your constructors:
    embeddings = HuggingFaceEmbeddings(model_name=str(mpnet_local))
    cross_encoder = HuggingFaceCrossEncoder(model_name=str(reranker_local))
    vector_store = FAISS.from_documents(docs, embeddings)
    print("Total vectors in vector_store:",vector_store.index.ntotal)

    bm25_retriever = BM25Retriever.from_texts(
        [doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs],
        k=5
    )
    vect_retriever = vector_store.as_retriever(k=20)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vect_retriever],
        weights=[0.5, 0.5],  # adjust to balance keyword vs. semantic
        c=60                  # RRF constant
    )

    #cross_encoder = HuggingFaceCrossEncoder(model_name="models/BAAI_bge_reranker")

    reranker = CrossEncoderReranker(model=cross_encoder, top_n=10)

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever
    )
    return retriever

docs = load_documents("knowledge_base")
retriever = setup_retriever(docs)