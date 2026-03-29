import os
from rag.loader import load_job_descriptions_from_directory, split_documents
from rag.retriever import create_question_generator, create_retriever, create_vector_store, load_vector_store


def setup_interview_rag_from_directory(
    job_descriptions_dir: str,
    persist_directory: str = "./chroma_db"
) -> dict:
    """Build or load all RAG components from a directory of job descriptions."""
    if os.path.exists(persist_directory):
        vector_store = load_vector_store(persist_directory)
        chunks = [] # Not needed when loading
    else:
        docs = load_job_descriptions_from_directory(job_descriptions_dir)
        chunks = split_documents(docs)
        vector_store = create_vector_store(chunks, persist_directory)

    retriever = create_retriever(vector_store)
    question_generator = create_question_generator(retriever)

    return {
        "chunks": chunks,
        "vector_store": vector_store,
        "retriever": retriever,
        "question_generator": question_generator,
    }

