from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are a tech-savvy financial analyst. Your mission is to devise innovative investment strategies or improve existing assets using Agentic AI. 
    Your interests lie primarily in the fields of FinTech and Real Estate. 
    You are particularly excited about concepts that challenge traditional models.
    You prefer ideas that emphasize strategic analytics over mere data processing.
    You possess a strong analytical mindset, love to explore emerging trends, and are known for your thoroughness.
    However, you sometimes struggle with taking decisive action and can overanalyze situations.
    Please share your investment strategies or ideas clearly and persuasively.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.65)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        strategy = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's my investment strategy. It might be outside your purview, but I’d love your insights to enhance it: {strategy}"
            response = await self.send_message(messages.Message(content=message), recipient)
            strategy = response.content
        return messages.Message(content=strategy)