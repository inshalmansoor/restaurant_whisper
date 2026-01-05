from langchain.chat_models import init_chat_model
from langchain_classic.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from chat_model import chat_model
from datetime import datetime

class QueryRun:
    output_schema = None   # subclasses override this

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.output_schema:
            cls.output_parser = PydanticOutputParser(pydantic_object=cls.output_schema)
        else:
            cls.output_parser = None

    def execute_query(self, query, conversation_history="", context="", order="", customer_details="", booking="", current_complain="", parser=None):
        # Default parser comes from subclass
        parser = parser or self.__class__.output_parser

        # Prompt template
        final_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", self.human_message + ("\n\n{format_instructions}" if parser else ""))
        ])

        # Chain with model
        final_chain = final_prompt_template | chat_model
        if parser:
            final_chain = final_chain | parser

        # Inputs
        inputs = {
            "query": query,
            "conversation_history": conversation_history,
            "context": context,
            "order": order,
            "customer": customer_details,
            "booking": booking,
            "current_complaint": current_complain,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
        }
        if parser:
            inputs["format_instructions"] = parser.get_format_instructions()

        # Run
        answer = final_chain.invoke(inputs)
        return answer