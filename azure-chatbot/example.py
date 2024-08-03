# Library imports
import asyncio
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# Azure imports
from azure.ai.contentsafety.aio import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.cosmos import CosmosClient

# Semantic Kernel imports
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.functions.kernel_arguments import KernelArguments

# Configuration imports
from config import (
    AZURE_COSMOSDB_DATABASE_NAME,
    AZURE_COSMOSDB_CONTAINER_NAME,
    AZURE_COSMOSDB_ENDPOINT,
    AZURE_COSMOSDB_KEY,
    AZURE_CONTENT_SAFETY_ENDPOINT,
    AZURE_CONTENT_SAFETY_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

class ChatbotService:
    def __init__(self, loop, user_id):
        """
        Initialize the ChatbotService with event loop, user ID, and necessary Azure CosmosDB clients.
        """

        # Set up the event loop and thread pool executor for managing asynchronous tasks, 
        # allowing Azure Cosmos DB operations to be handled in an asynchronous environment
        self.loop = loop
        self.executor = ThreadPoolExecutor()

        # Store the user ID, which is used as the partition key (/userId) in Azure CosmosDB
        self.user_id = user_id

        # Initialize the Azure Cosmos DB client
        self.cosmos_client = CosmosClient(AZURE_COSMOSDB_ENDPOINT, credential=AZURE_COSMOSDB_KEY)

        # Initialize the Azure Cosmos DB database client
        self.database = self.cosmos_client.get_database_client(AZURE_COSMOSDB_DATABASE_NAME)

        # Initialize the Azure Cosmos DB container client
        self.container = self.database.get_container_client(AZURE_COSMOSDB_CONTAINER_NAME)

    async def init(self):
        """
        Initialize the Content Safety client and Semantic Kernel.
        """

        # Initialize the Azure Content Safety client
        self.content_safety_client = ContentSafetyClient(AZURE_CONTENT_SAFETY_ENDPOINT, AzureKeyCredential(AZURE_CONTENT_SAFETY_KEY))
        
        # Initialize the Semantic Kernel
        self.kernel = sk.Kernel()
        
        # Initialize the chat service for Azure OpenAI
        self.chat_service = AzureChatCompletion(
            service_id='chat_service', 
            deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, 
            endpoint=AZURE_OPENAI_ENDPOINT, 
            api_key=AZURE_OPENAI_KEY
        )
        
        # Add the chat service to the Semantic Kernel
        self.kernel.add_service(self.chat_service)

        # Define the prompt template configuration for the chatbot
        self.prompt_template_config = PromptTemplateConfig(
            template="""ChatBot can have a conversation with you about any topic.
                        It can give explicit instructions or say 'I don't know' if it does not have an answer.
                        {{$history}} 
                        User: {{$user_message}}
                        ChatBot: """,
            name='chat_prompt_template',
            template_format='semantic-kernel',
            input_variables=[
                InputVariable(name='user_message', description='The user message.', is_required=True),
                InputVariable(name='history', description='The conversation history', is_required=True),
            ],
            execution_settings=sk_oai.OpenAIChatPromptExecutionSettings(
                service_id='chat_service',
                ai_model_id='gpt-3.5-turbo',
                max_tokens=500,
                temperature=0.7
            )
        )

        # Add the chat function to the Semantic Kernel
        self.chat_function = self.kernel.add_function(
            function_name="chat_function",
            plugin_name="chat_plugin",
            prompt_template_config=self.prompt_template_config,
        )
        return self

    async def analyze_text(self, text):
        """
        Analyze the input text for safety using Azure Content Safety.
        """

        # Create a request with the input text to be analyzed
        request = AnalyzeTextOptions(text=text)
        try:
            # Send the request to the Azure Content Safety client and await the response
            response = await self.content_safety_client.analyze_text(request)

            # Get the analysis results for different categories
            results = {
                'hate': next((item for item in response.categories_analysis if item.category == TextCategory.HATE), None),
                'self_harm': next((item for item in response.categories_analysis if item.category == TextCategory.SELF_HARM), None),
                'sexual': next((item for item in response.categories_analysis if item.category == TextCategory.SEXUAL), None),
                'violence': next((item for item in response.categories_analysis if item.category == TextCategory.VIOLENCE), None),
            }

            # Print content safety analysis results
            print("\n<-- Content Safety Analysis Results -->")
            for category, result in results.items():
                if result:
                    print(f"{category.capitalize()} severity: {result.severity}")
            print("<-- End of Content Safety Analysis Results -->\n")

            # Define a threshold for the text to be considered unsafe
            threshold = 2

            # Based on the threshold, determine if the text is safe
            is_safe = not any(result and result.severity >= threshold for result in results.values())
            return is_safe, results

        except HttpResponseError as e:
            # Handle any HTTP response errors that occur during the request
            print(f"Failed to analyze text. Error: {e}")

    async def chat_with_kernel(self, user_message, history):
        """
        Interact with the chatbot using the Semantic Kernel and provided history.
        """

        # Create arguments for the chat function using the user message and conversation history stored in Azure Cosmos DB
        arguments = KernelArguments(user_message=user_message, history=history)

        # Invoke the chat function in the Semantic Kernel with the provided arguments and await the response
        response = await self.kernel.invoke(self.chat_function, arguments)

        # Return the chatbot's response
        return response

    async def store_interaction(self, user_message, chat_response):
        """
        Store the user interaction with the chatbot in Azure Cosmos DB.
        """

        # Run the _store_interaction_sync method in an asynchronous execution environment
        await self.loop.run_in_executor(self.executor, self._store_interaction_sync, user_message, chat_response)

    def _store_interaction_sync(self, user_message, chat_response):
        """
        Synchronously store the interaction in Azure Cosmos DB.
        """

        # Get the current time in UTC
        current_time = datetime.now(timezone.utc)

        # Upsert (insert or update) the interaction data into the Cosmos DB container
        self.container.upsert_item({
            'id': str(current_time.timestamp()),  # Use the current timestamp as a unique ID
            'user_message': user_message,  # Store the user message
            'bot_response': chat_response,  # Store the chatbot response
            'timestamp': current_time.isoformat(),  # Store the timestamp in ISO format
            'userId': self.user_id  # Store the user ID for partition key
        })

    async def load_historical_context(self):
        """
        Load the user's chat history from Azure Cosmos DB.
        """

        # Run the _load_historical_context_sync method in an asynchronous execution environment
        return await self.loop.run_in_executor(self.executor, self._load_historical_context_sync)

    def _load_historical_context_sync(self):
        """
        Synchronously load the user's chat history from Azure Cosmos DB.
        """

        # Define the query to select items for the current user, ordered by timestamp
        query = "SELECT * FROM c WHERE c.userId = @userId ORDER BY c.timestamp DESC"
        parameters = [{"name": "@userId", "value": self.user_id}]
    
        # Execute the query and retrieve the items
        items = list(self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

        # Include only the last 5 conversations
        history_items = items[:5]

        # Format the conversion history
        return "\n".join([f"User: {item['user_message']}\nChatBot: {item['bot_response']}" for item in history_items])

    async def clear_historical_context(self):
        """
        Clear the user's chat history from Azure Cosmos DB.
        """

        # Run the _clear_historical_context_sync method in an asynchronous execution environment
        await self.loop.run_in_executor(self.executor, self._clear_historical_context_sync)

    def _clear_historical_context_sync(self):
        """
        Synchronously clear the user's chat history from Azure Cosmos DB.
        """

        # Define the query to select items for the current user
        query = "SELECT * FROM c WHERE c.userId = @userId"
        parameters = [{"name": "@userId", "value": self.user_id}]

        # Execute the query and retrieve the items
        items = list(self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    
        # Clear the chat history by deleting all items for the current user
        for item in items:
            self.container.delete_item(item, partition_key=self.user_id)

async def main():
    """
    Main function to run the chatbot service.
    """

    # Enter the User ID to join the chat.
    # A conversation history is stored based on the user ID.
    user_id = input("User ID: ")

    # Get the event loop to allow synchronous operations in Azure Cosmos DB for NoSQL to run asynchronously.
    loop = asyncio.get_running_loop()

    # Initialize the ChatbotService with the event loop and user ID
    chatbot_service = await ChatbotService(loop, user_id).init()

    # Main loop for the interaction with the user
    while True:
        user_message = input("You: ")
        # Exit the chat loop if the user types 'exit'
        if user_message.lower() == 'exit':
            break

        elif user_message.lower() == 'history':
            # Load and print the chat history if the user types 'history'
            history = await chatbot_service.load_historical_context()
            print(f"\n<-- Chat history of user ID: {user_id} -->")
            print(history)
            print(f"<-- End of chat history of user ID: {user_id} -->\n")

        elif user_message.lower() == 'clear':
            # Clear the chat history if the user types 'clear'
            await chatbot_service.clear_historical_context()
            print("Chat history cleared.")

        else:
            # Analyze the text for safety
            is_safe, _ = await chatbot_service.analyze_text(user_message)

            if is_safe:
                # Load chat history and interact with the chatbot if the message is safe
                history = await chatbot_service.load_historical_context()
                chat_response = await chatbot_service.chat_with_kernel(user_message, history)
                print("Chatbot:", chat_response)

                # Store the interaction in Cosmos DB
                await chatbot_service.store_interaction(user_message, str(chat_response))
            else:
                # Inform the user if their message is not safe
                print("Chatbot: Your message is not safe to process. Please rephrase and try again.")

if __name__ == "__main__":
    asyncio.run(main())