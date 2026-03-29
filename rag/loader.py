import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

from config import settings


def load_job_description(path: str) -> List[Document]:
	"""Load a job description file from txt, md, pdf, or docx."""
	file_path = Path(path)
	suffix = file_path.suffix.lower()

	if suffix in {".txt", ".md"}:
		loader = TextLoader(str(file_path), encoding="utf-8")
	elif suffix == ".pdf":
		loader = PyPDFLoader(str(file_path))
	elif suffix == ".docx":
		loader = Docx2txtLoader(str(file_path))
	else:
		raise ValueError(f"Unsupported job description format: {suffix}")

	return loader.load()


def create_docs_from_text(text: str, source: str = "inline_job_description") -> List[Document]:
	"""Create a single LangChain document from raw text."""
	return [Document(page_content=text, metadata={"source": source})]


def load_job_descriptions_from_directory(path: str) -> list[Document]:
	"""Load all .txt files from a directory into a list of documents."""
	if not os.path.isdir(path):
		raise ValueError(f"Job descriptions directory not found: {path}")

	docs: list[Document] = []
	for filename in sorted(os.listdir(path)):
		if filename.endswith(".txt"):
			file_path = os.path.join(path, filename)
			with open(file_path, "r", encoding="utf-8") as f:
				docs.append(Document(page_content=f.read(), metadata={"source": filename}))

	if not docs:
		raise ValueError(f"No .txt files found in directory: {path}")

	return docs



def split_documents(
	docs: List[Document],
	chunk_size: int | None = None,
	chunk_overlap: int | None = None,
) -> List[Document]:
	"""Split source docs into chunks for vector search."""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size or settings.chunk_size,
		chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
	)
	return splitter.split_documents(docs)
