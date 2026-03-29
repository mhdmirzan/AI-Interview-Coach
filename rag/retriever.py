import re
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings


def create_vector_store(chunks: List[Document], persist_directory: str = "./chroma_db") -> Chroma:
    """Create and persist a Chroma vector store."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


def load_vector_store(persist_directory: str = "./chroma_db") -> Chroma:
    """Load an existing Chroma vector store."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def create_retriever(vector_store: Chroma, k: int | None = None):
    """Build a retriever configured with top-k similarity search."""
    k_value = k or settings.retriever_k
    return vector_store.as_retriever(search_kwargs={"k": k_value})



def _format_docs(docs: List[Document]) -> str:
	if not docs:
		return "No relevant job description context found."
	return "\n\n".join(doc.page_content for doc in docs)


def create_question_generator(retriever):
	"""Create a RAG-based chain that generates one interview question."""
	prompt = ChatPromptTemplate.from_template(
		"""You are an expert technical interviewer.

Use the job description context to ask exactly one interview question.

Topic: {topic}
Difficulty: {difficulty}
Recent questions: {previous_questions}

Job description context:
{context}

Rules:
- Ask one clear question only.
- Avoid repeating recent questions.
- Match the requested difficulty.
"""
	)

	llm = ChatOpenAI(
		model=settings.model_name,
		temperature=settings.temperature,
		api_key=settings.openai_api_key,
	)

	chain = (
		{
			"topic": itemgetter("topic"),
			"difficulty": itemgetter("difficulty"),
			"previous_questions": itemgetter("previous_questions"),
			"context": itemgetter("topic") | RunnableLambda(lambda q: retriever.invoke(q)) | _format_docs,
		}
		| prompt
		| llm
		| StrOutputParser()
	)

	return chain
