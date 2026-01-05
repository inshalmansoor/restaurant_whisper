from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    api_key="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",                  # Your API key
    base_url="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4.1-mini"
)