from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from config import settings

INTERVIEWER_SYSTEM_PROMPT = """You are an expert technical interviewer.

Your role:
- Ask one clear, focused question at a time
- Reference previous answers when relevant
- Build on the conversation naturally
- Be professional but encouraging

Interview type: {interview_type}
Position level: {level}
Focus area: {focus_area}

Remember: You have access to the full conversation history.
Use it to avoid repeating questions and to ask follow-ups.
"""

def create_interviewer_chain_with_memory(memory):
    """Create interviewer chain with conversation memory."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTERVIEWER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

    return prompt | llm | StrOutputParser()


def create_interviewer_with_history():
    """Create an interviewer chain that stores chat history by session_id."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INTERVIEWER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )
    base_chain = prompt | llm | StrOutputParser()

    store: dict[str, BaseChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )