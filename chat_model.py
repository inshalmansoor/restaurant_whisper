from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    api_key="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",                  # Your API key
    base_url="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4.1-mini"
)