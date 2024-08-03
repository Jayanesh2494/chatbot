# Azure Cosmos DB settings
AZURE_COSMOSDB_ACCOUNT_NAME = 'your-cosmosdb-account-name' # 'safetychatbot-storage'
AZURE_COSMOSDB_DATABASE_NAME = 'your-cosmosdb-database-name' # 'safetychatbot-database'
AZURE_COSMOSDB_CONTAINER_NAME = 'your-cosmosdb-container-name' # 'safetychatbot-container'
AZURE_COSMOSDB_ENDPOINT = f'https://{AZURE_COSMOSDB_ACCOUNT_NAME.lower()}.documents.azure.com:443/'
AZURE_COSMOSDB_KEY = 'your-cosmosdb-key'

# Azure Content Safety settings
AZURE_CONTENT_SAFETY_NAME = 'your-content-safety-name' # 'safetychatbot-contentsafety'
AZURE_CONTENT_SAFETY_ENDPOINT = f'https://{AZURE_CONTENT_SAFETY_NAME.lower()}.cognitiveservices.azure.com/'
AZURE_CONTENT_SAFETY_KEY = 'your-content-safety-key'

# Azure OpenAI settings
AZURE_OPENAI_NAME = 'safetychatbot-openai'
AZURE_OPENAI_ENDPOINT = f'https://{AZURE_OPENAI_NAME.lower()}.openai.azure.com/'
AZURE_OPENAI_KEY = 'your-openai-key'
AZURE_OPENAI_API_VERSION = 'your-API-version' # '2024-02-15-preview'
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = 'your_chat_deployment_name' # 'gpt-35-turbo'