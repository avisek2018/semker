import asyncio
import json
from typing import List, Dict, Annotated
import time
import logging
from mcp.client.session import ClientSession
import aiohttp
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.connectors.ai import FunctionChoiceBehavior

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration (replace with your actual credentials)
AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_API_KEY = "YOUR_AZURE_OPENAI_API_KEY"
AZURE_TRANSLATOR_ENDPOINT = "YOUR_AZURE_TRANSLATOR_ENDPOINT"
AZURE_TRANSLATOR_API_KEY = "YOUR_AZURE_TRANSLATOR_API_KEY"
AZURE_TRANSLATOR_REGION = "YOUR_AZURE_TRANSLATOR_REGION"
MCP_SERVER_URL = "http://localhost:8000"  # MCP server URL
MCP_LIST_FILES_ENDPOINT = "/tools/list_files"  # Adjusted for FastMCP
MCP_CREATE_FILE_ENDPOINT = "/tools/create_file"  # Adjusted for FastMCP


class LanguageDetectionPlugin:
    """A plugin for detecting language using Azure Translator."""

    def __init__(self):
        try:
            self.client = TextTranslationClient(
                endpoint=AZURE_TRANSLATOR_ENDPOINT,
                credential=AzureKeyCredential(AZURE_TRANSLATOR_API_KEY),
                region=AZURE_TRANSLATOR_REGION
            )
            logger.debug("Initialized Azure Translator client for language detection")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Translator client: {str(e)}")
            raise

    @kernel_function(
        name="detect_language",
        description="Detect the language of the input text."
    )
    async def detect_language(
        self,
        text: Annotated[str, "The text to detect language for"]
    ) -> Annotated[str, "Detected language code and name"]:
        """Detect the language of the input text using Azure Translator."""
        try:
            result = await asyncio.to_thread(
                self.client.translate,
                body=[{"text": text}],
                to_language=["en"]
            )
            if result and result[0].detected_language:
                language = result[0].detected_language
                logger.debug(f"Detected language for '{text}': {language.language} (Confidence: {language.score})")
                return f"Detected language: {language.language} (Confidence: {language.score:.2f})"
            logger.warning(f"Unable to detect language for '{text}'")
            return "Unable to detect language."
        except Exception as e:
            logger.error(f"Error detecting language for '{text}': {str(e)}")
            return f"Error detecting language: {str(e)}"

class TranslationPlugin:
    """A plugin for translating text to English using Azure Translator."""

    def __init__(self):
        try:
            self.client = TextTranslationClient(
                endpoint=AZURE_TRANSLATOR_ENDPOINT,
                credential=AzureKeyCredential(AZURE_TRANSLATOR_API_KEY),
                region=AZURE_TRANSLATOR_REGION
            )
            self.last_translation = None  # Store the last translation for saving
            logger.debug("Initialized Azure Translator client for translation")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Translator client: {str(e)}")
            raise

    @kernel_function(
        name="translate_to_english",
        description="Translate the input text to English."
    )
    async def translate_to_english(
        self,
        text: Annotated[str, "The text to translate to English"]
    ) -> Annotated[str, "Translated text in English"]:
        """Translate the input text to English using Azure Translator."""
        try:
            result = await asyncio.to_thread(
                self.client.translate,
                body=[{"text": text}],
                to_language=["en"]
            )
            if result and result[0].translations:
                translation = result[0].translations[0].text
                self.last_translation = translation
                logger.debug(f"Translated '{text}' to English: {translation}")
                return f"Translated to English: {translation}"
            logger.warning(f"Unable to translate text: '{text}'")
            return "Unable to translate text."
        except Exception as e:
            logger.error(f"Error translating text: '{str(e)}")
            return f"Error translating text: {str(e)}"

class TextFilePlugin:
    """A plugin for interacting with the MCP server for local text file operations."""

    def __init__(self):
        self.http_session = None
        self.plugins = {}  # To store reference to other plugins

    async def initialize_client(self):
        """Initialize the aiohttp ClientSession."""
        try:
            self.http_session = aiohttp.ClientSession()
            logger.debug("Initialized aiohttp ClientSession for MCP")
        except Exception as e:
            logger.error(f"Failed to initialize aiohttp ClientSession: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up the aiohttp ClientSession."""
        try:
            if self.http_session:
                await self.http_session.close()
                self.http_session = None
                logger.debug("Cleaned up aiohttp ClientSession")
        except Exception as e:
            logger.error(f"Failed to cleanup aiohttp ClientSession: {str(e)}")

    @kernel_function(
        name="list_tools",
        description="List available tools on the MCP server."
    )
    async def list_tools(self) -> Annotated[List[str], "List of available tool names"]:
        """List available tools from the MCP server."""
        try:
            if self.http_session is None:
                await self.initialize_client()
            tools = ["list_files", "create_file"]
            logger.debug(f"Listed tools (hardcoded): {tools}")
            return tools
        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            return [f"Error listing tools: {str(e)}"]

    @kernel_function(
        name="invoke_tool",
        description="Invoke a tool on the MCP server."
    )
    async def invoke_tool(
        self,
        tool_name: Annotated[str, "Name of the tool to invoke (e.g., 'list_files', 'create_file')"],
        file_name: Annotated[str, "Name for the new text file (optional for create_file)"] = None
    ) -> Annotated[Dict, "Result of the tool invocation"]:
        """Invoke a specified tool from the MCP server using HTTP requests."""
        try:
            if self.http_session is None:
                await self.initialize_client()
            valid_tools = ["list_files", "create_file"]
            if tool_name not in valid_tools:
                logger.error(f"Tool '{tool_name}' not found on MCP server")
                return {"status": "error", "message": f"Tool '{tool_name}' not found on MCP server"}

            if tool_name == "list_files":
                async with self.http_session.get(f"{MCP_SERVER_URL}{MCP_LIST_FILES_ENDPOINT}") as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"List files result: {result}")
                        return result
                    else:
                        error_msg = await response.text()
                        logger.error(f"Failed to list files: HTTP {response.status} - {error_msg}")
                        return {
                            "status": "error",
                            "message": f"Failed to list files: HTTP {response.status} - {error_msg}. "
                                       f"Please ensure the MCP server is running and the endpoint '{MCP_LIST_FILES_ENDPOINT}' is correct."
                        }
            elif tool_name == "create_file":
                if not file_name:
                    file_name = f"Translation_{time.strftime('%Y%m%d%H%M%S')}.txt"
                translation = self.plugins.get("translation", {}).last_translation
                if not translation:
                    logger.warning("No recent translation to save")
                    return {"status": "error", "message": "No recent translation to save. Please translate something first."}
                data = [{"translation": translation}]
                async with self.http_session.post(
                    f"{MCP_SERVER_URL}{MCP_CREATE_FILE_ENDPOINT}",
                    params={"file_name": file_name},  # Send file_name as query parameter
                    json=data  # Send data as the body
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Create file result: {result}")
                        return result
                    else:
                        error_msg = await response.text()
                        logger.error(f"Failed to create file: HTTP {response.status} - {error_msg}")
                        return {
                            "status": "error",
                            "message": f"Failed to create file: HTTP {response.status} - {error_msg}. "
                                       f"Please ensure the MCP server is running and the endpoint '{MCP_CREATE_FILE_ENDPOINT}' is correct."
                        }
            else:
                logger.error(f"Unsupported tool: {tool_name}")
                return {"status": "error", "message": f"Unsupported tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Failed to invoke tool '{tool_name}': {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to invoke tool '{tool_name}': {str(e)}. "
                           f"Please ensure the MCP server is running at '{MCP_SERVER_URL}'."
            }

class LanguageAgent:
    """An agent that handles language detection, translation, and file saving requests."""

    def __init__(self):
        self.kernel = Kernel()
        self.kernel.add_service(AzureChatCompletion(
            service_id="chat",
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name="gpt-4.1-mini"
        ))
        self.language_detection_plugin = LanguageDetectionPlugin()
        self.kernel.add_plugin(self.language_detection_plugin, plugin_name="language_detection")
        self.translation_plugin = TranslationPlugin()
        self.kernel.add_plugin(self.translation_plugin, plugin_name="translation")
        self.text_file_plugin = TextFilePlugin()
        self.kernel.add_plugin(self.text_file_plugin, plugin_name="text_file")
        self.text_file_plugin.plugins = {
            "translation": self.translation_plugin
        }  # Provide access to translation plugin
        self.state = "awaiting_input"
        self.last_input = None
        self.agent = ChatCompletionAgent(
            service=AzureChatCompletion(
                service_id="chat",
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                deployment_name="gpt-4.1-mini"
            ),
            name="LanguageAssistant",
            instructions="""
You are a helpful language assistant. Your goal is to assist users in detecting language, translating to English, and saving translations using Azure Cognitive Services and the MCP file server.

Follow these guidelines:
1. Ask the user for input text to process (e.g., 'Please provide text to process, e.g., "Bonjour le monde".').
2. Store the input text and prompt for one of three options: 1. Detect language, 2. Translate to English, 3. Save the translation to local file system.
3. If the user chooses '1', call the `detect_language` function and display the detected language.
4. If the user chooses '2', call the `translate_to_english` function and display the translation.
5. If the user chooses '3', call the `invoke_tool` function with `create_file` to save the most recent translation, then ask if they want to list files.
6. If the user chooses to list files ('list files'), call the `invoke_tool` function with `list_files`.
7. After each action, prompt for the next option or new text (e.g., 'What would you like to do? 1. Detect language, 2. Translate to English, 3. Save translation, or provide new text.').
8. Format responses clearly, e.g., 'Detected language: French (fr)', 'Translated to English: Hello world', 'Saved to Translation_202508011230.txt'.
9. For saving, use a unique file name (e.g., 'Translation_202508011230.txt') via the MCP server.
10. If the user provides text without selecting an option, store it and prompt for an option.

**Error Handling**:
- If an API call fails, return a user-friendly error message (e.g., 'Failed to detect language. Please try again.') and suggest retrying.
- If no translation is available to save, inform the user (e.g., 'No recent translation to save. Translate something first.').
- If the user provides invalid input, prompt for clarification (e.g., 'Please provide valid text or select an option: 1, 2, or 3.').
- For file operations, rely on the MCP server's `create_file` and `list_files` tools defined in files_mcp.py.
""",
            plugins=[self.language_detection_plugin, self.translation_plugin, self.text_file_plugin],
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )

    async def initialize(self):
        """Initialize async components of the agent."""
        try:
            await self.text_file_plugin.initialize_client()
            logger.debug("Initialized LanguageAgent")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up async resources."""
        try:
            await self.text_file_plugin.cleanup()
            logger.debug("Cleaned up LanguageAgent")
        except Exception as e:
            logger.error(f"Failed to cleanup: {str(e)}")

    async def process_message(self, message: str) -> str:
        """Process user messages and handle options directly."""
        try:
            logger.debug(f"Processing message: '{message}', State: {self.state}, Last input: {self.last_input}")

            # Handle 'exit' command
            if message.lower() == "exit":
                logger.debug("Exit command received")
                return "Exiting Language Assistant. Goodbye!"

            # Handle text input
            if not message.isdigit() and message.lower() not in ["list files"]:
                self.last_input = message.strip()
                self.state = "awaiting_option"
                logger.debug(f"Stored input text: {self.last_input}")
                return f"Received text: '{self.last_input}'.\nWhat would you like to do? 1. Detect language, 2. Translate to English, 3. Save translation, or 'list files' to see saved files."

            # Handle options
            if self.state == "awaiting_option":
                if message == "1":
                    if not self.last_input:
                        logger.warning("No input text provided for language detection")
                        return "Please provide text to detect language, e.g., 'Bonjour le monde'."
                    result = await self.language_detection_plugin.detect_language(self.last_input)
                    logger.debug(f"Language detection result: {result}")
                    return f"{result}\nWhat would you like to do next? 1. Detect language, 2. Translate to English, 3. Save translation, or 'list files'."
                elif message == "2":
                    if not self.last_input:
                        logger.warning("No input text provided for translation")
                        return "Please provide text to translate, e.g., 'Bonjour le monde'."
                    result = await self.translation_plugin.translate_to_english(self.last_input)
                    logger.debug(f"Translation result: {result}")
                    return f"{result}\nWhat would you like to do next? 1. Detect language, 2. Translate to English, 3. Save translation, or 'list files'."
                elif message == "3":
                    if not self.translation_plugin.last_translation:
                        logger.warning("No translation available to save")
                        return "No recent translation to save. Please translate something first."
                    result = await self.text_file_plugin.invoke_tool(tool_name="create_file")
                    logger.debug(f"File save result: {result}")
                    if result.get("status") != "error":
                        formatted = f"Saved translation to {result.get('file_name', 'Translation.txt')}.\n"
                    else:
                        formatted = f"Failed to save translation: {result.get('message', 'Unknown error')}.\n"
                    formatted += "What would you like to do next? 1. Detect language, 2. Translate to English, 3. Save translation, or 'list files'."
                    return formatted
                elif message.lower() == "list files":
                    result = await self.text_file_plugin.invoke_tool(tool_name="list_files")
                    logger.debug(f"List files result: {result}")
                    if isinstance(result, list):
                        file_names = [f["name"] for f in result]
                        formatted = f"Available files: {', '.join(file_names) if file_names else 'No files found'}\n"
                    else:
                        formatted = f"Failed to list files: {result.get('message', 'Unknown error')}.\n"
                    formatted += "What would you like to do next? 1. Detect language, 2. Translate to English, 3. Save translation, or 'list files'."
                    return formatted
                else:
                    logger.warning(f"Invalid option: {message}")
                    return "Please select a valid option: 1. Detect language, 2. Translate to English, 3. Save translation, or 'list files'."

            # Fallback for unexpected state or input
            logger.warning(f"Unexpected state: {self.state} or input: {message}")
            return "Please provide text to process or select an option: 1. Detect language, 2. Translate to English, 3. Save translation, or 'list files'."
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            return f"Error processing request: {str(e)}. Please provide text to process or select an option: 1, 2, or 3."

async def main():
    agent = LanguageAgent()
    try:
        await agent.initialize()
        print("""
Welcome to the Language Assistant! Please provide text to process, e.g., 'Bonjour le monde'.
Then select an option: 1. Detect language, 2. Translate to English, 3. Save the translation to a file.
You can also list available files with 'list files'. Type 'exit' to quit.
""")
        
        while True:
            user_input = input("You: ")
            response = await agent.process_message(user_input)
            print(f"Assistant: {response}")
            if response == "Exiting Language Assistant. Goodbye!":
                break
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        print(f"Error in main loop: {str(e)}")
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())