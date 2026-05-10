#LLM Factory for model agnostic usage using langchain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from config import settings

#langchain has a core interface a base chat model
#using that interface allows us to swap LLMs with zero rewrite
#we mainly want to use .invoke(message) and .aiinvoke(message) for async calls
#gets temp, provider, and model on use
def get_llm(
    temperature: float = 0.0,
    provider: str | None = None,
    model: str | None = None,
    #returns a langchain model using information from BaseChatModel
) -> BaseChatModel:
    provider = provider or settings.llm_provider
    model = model or settings.llm_model

#type hinting so i dont forget again
#match with common providers (will add ollama once i figure out how to setup on cloud)
    match provider:
        case "google":
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=settings.google_api_key,
            )
        case "openai":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=settings.openai_api_key,
            )
        case "anthropic":
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=settings.anthropic_api_key,
            )
        case _:
            raise ValueError(f"Unknown provider: {provider}")