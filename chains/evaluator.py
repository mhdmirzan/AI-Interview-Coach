from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings


class AnswerFeedback(BaseModel):
	score: int = Field(ge=1, le=10)
	understanding: str = Field(description="Short evaluation summary")
	strengths: List[str] = Field(default_factory=list)
	improvements: List[str] = Field(default_factory=list)
	follow_up_question: str | None = None


class InterviewReport(BaseModel):
	overall_score: float = Field(ge=1, le=10)
	recommendation: Literal["hire", "lean_hire", "no_hire"]
	summary: str
	strengths: List[str] = Field(default_factory=list)
	areas_to_improve: List[str] = Field(default_factory=list)
	suggested_topics_to_study: List[str] = Field(default_factory=list)


def create_evaluator_simple():
	"""Return a chain that evaluates a single answer with structured feedback."""
	prompt = ChatPromptTemplate.from_template(
		"""You are a senior interview evaluator.

Question:
{question}

Candidate level: {level}

Candidate answer:
{answer}

Evaluate this answer and provide constructive feedback.
"""
	)

	llm = ChatOpenAI(
		model=settings.model_name,
		temperature=0.2,
		api_key=settings.openai_api_key,
	)
	structured_llm = llm.with_structured_output(AnswerFeedback)
	return prompt | structured_llm


def create_report_generator():
	"""Return a chain that creates the final interview report."""
	prompt = ChatPromptTemplate.from_template(
		"""You are preparing a final interview panel report.

Position: {position}
Level: {level}
Interview type: {interview_type}

Transcript:
{transcript}

Per-question scores: {scores}

Generate a fair, actionable final report.
"""
	)

	llm = ChatOpenAI(
		model=settings.model_name,
		temperature=0.2,
		api_key=settings.openai_api_key,
	)
	structured_llm = llm.with_structured_output(InterviewReport)
	return prompt | structured_llm
