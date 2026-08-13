"""
LLM Factory

Understands provider, able to access settins
Use to create any agent to abstract away the provider in use
"""

from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, SecretStr

from agenticresume.settings import Settings


#cool typevar usage, since were removing the llm invoke from agents, we need to make it schema agnostic
#serves as a "whatever fills me in" typevar, so we can use it to type the output of the llm invoke
SchemaT = TypeVar("SchemaT", bound=BaseModel)

def _require(key: SecretStr | None, provider: str) -> SecretStr:
    """Return the active provider's key, guaranteed present by settings.

    Returns the SecretStr unchanged (langchain clients accept SecretStr and
    unwrap it themselves), so the plaintext is never handled here."""

    if key is None:
        raise ValueError(f"missing api key for provider {provider!r}")
    return key

def build_chat_model(settings: Settings) -> BaseChatModel:
    """Construct the configured chat model"""

    model = settings.llm_model
    temperature= settings.llm_temperature


    match settings.llm_provider:
        case "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=_require(settings.google_api_key, "google"),
            )
        
        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=_require(settings.openai_api_key, "openai"),
            )
        
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            # langchain-anthropic's typing is incomplete under pyright (aliases
            # model->model_name, marks defaulted fields required); runtime is fine.
            return ChatAnthropic(  # pyright: ignore[reportCallIssue]
                model=model,  # pyright: ignore[reportCallIssue]
                temperature=temperature,
                api_key=_require(settings.anthropic_api_key, "anthropic"),
                max_tokens=4096,  # pyright: ignore[reportCallIssue]  # anthropic requires an explicit cap
            )
        
        case "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=model,  # type: ignore[call-arg]  # groq aliases model->model_name; runtime is fine
                temperature=temperature,
                api_key=_require(settings.groq_api_key, "groq"),
            )
        case _:  # pragma: no cover - Literal makes this exhaustive
            raise ValueError(f"unsupported provider: {settings.llm_provider}")

def build_extractor(settings: Settings, schema: type[BaseModel]) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
    """A chat model constrained to emit `schema`."""

    #with_structured_output guarantees that the model explicitly reads the schema given and uses that for output
    #handy since it also parses it back into the wire schema
    return build_chat_model(settings).with_structured_output(schema)

#typevar allows us to use this function with any schema, and it will return the correct type
async def invoke_structured(settings: Settings,schema: type[SchemaT],system_prompt: str,user_content: str,) -> SchemaT:
    """Call the model with a system prompt and user content, return a `schema` instance."""
    model = build_extractor(settings, schema)
    result = await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    )
    if not isinstance(result, schema):
        raise TypeError(f"model returned {type(result).__name__}, expected {schema.__name__}")
    return result