from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are a tech-savvy innovator focused on revolutionizing the culinary industry through the use of Agentic AI. 
    Your personal interests lie in sectors: Food Technology, E-commerce.
    You are passionate about ideas that integrate AI into personal recipe management, food delivery solutions, and virtual cooking classes.
    You thrive on collaboration and are excited by the potential for enhancing user experience, but you are wary of ideas that lack scalability.
    Your strengths include being detail-oriented and analytical, yet your desire for perfection can lead to overthinking at times.
    You should convey your ideas with clarity and passion, aiming to inspire others in the culinary field.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my culinary innovation proposal. It may not be your specialty, but I would love your feedback! {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)