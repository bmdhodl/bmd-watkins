"""
LLM Client for Watkins
Supports both cloud (Anthropic Claude) and local (Ollama) models
"""

import logging
from typing import List, Dict, Optional
import time
import requests
from anthropic import Anthropic


class LLMClient:
    """LLM Client with cloud/local model switching"""

    def __init__(
        self,
        mode: str = "hybrid",
        cloud_provider: str = "anthropic",
        cloud_model: str = "claude-3-5-sonnet-20241022",
        cloud_api_key: Optional[str] = None,
        local_host: str = "http://localhost:11434",
        local_model: str = "phi3.5",
        max_tokens: int = 150,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLM Client

        Args:
            mode: Mode (cloud, local, hybrid)
            cloud_provider: Cloud provider (anthropic, openai)
            cloud_model: Cloud model name
            cloud_api_key: API key for cloud provider
            local_host: Ollama host URL
            local_model: Local model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt for context
        """
        self.mode = mode
        self.cloud_provider = cloud_provider
        self.cloud_model = cloud_model
        self.local_host = local_host
        self.local_model = local_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or "You are a helpful voice assistant."

        self.logger = logging.getLogger(__name__)

        # Initialize cloud client
        if mode in ["cloud", "hybrid"]:
            if cloud_provider == "anthropic":
                if not cloud_api_key:
                    self.logger.warning("No Anthropic API key provided")
                    self.cloud_client = None
                else:
                    try:
                        self.cloud_client = Anthropic(api_key=cloud_api_key)
                        self.logger.info(f"Anthropic client initialized: {cloud_model}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize Anthropic client: {e}")
                        self.cloud_client = None
            else:
                self.logger.warning(f"Unsupported cloud provider: {cloud_provider}")
                self.cloud_client = None
        else:
            self.cloud_client = None

        # Test local connection
        if mode in ["local", "hybrid"]:
            if self._test_ollama_connection():
                self.logger.info(f"Ollama connected: {local_host} ({local_model})")
            else:
                self.logger.warning(f"Could not connect to Ollama at {local_host}")

    def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.local_host}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama connection test failed: {e}")
            return False

    def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        prefer_local: bool = False
    ) -> str:
        """
        Generate response using LLM

        Args:
            prompt: User prompt
            conversation_history: List of previous messages
            prefer_local: Prefer local model if available

        Returns:
            Generated response text
        """
        if conversation_history is None:
            conversation_history = []

        # Decide which model to use
        use_cloud = self._should_use_cloud(prefer_local)

        if use_cloud and self.cloud_client:
            return self._generate_cloud(prompt, conversation_history)
        else:
            return self._generate_local(prompt, conversation_history)

    def _should_use_cloud(self, prefer_local: bool) -> bool:
        """Determine whether to use cloud or local model"""
        if self.mode == "cloud":
            return True
        elif self.mode == "local":
            return False
        elif self.mode == "hybrid":
            # In hybrid mode, use cloud unless prefer_local is set
            if prefer_local:
                return False
            return self.cloud_client is not None
        return False

    def _generate_cloud(
        self,
        prompt: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate response using cloud API (Anthropic)"""
        if not self.cloud_client:
            self.logger.error("Cloud client not available, falling back to local")
            return self._generate_local(prompt, conversation_history)

        try:
            start_time = time.time()

            # Build messages
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": prompt})

            # Call Claude API
            response = self.cloud_client.messages.create(
                model=self.cloud_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages
            )

            response_text = response.content[0].text
            generation_time = time.time() - start_time

            self.logger.info(
                f"Cloud response generated in {generation_time:.2f}s "
                f"({len(response_text)} chars)"
            )
            self.logger.debug(f"Response: '{response_text}'")

            return response_text

        except Exception as e:
            self.logger.error(f"Cloud generation failed: {e}")
            # Fallback to local if available
            if self.mode == "hybrid":
                self.logger.info("Falling back to local model")
                return self._generate_local(prompt, conversation_history)
            return "I'm sorry, I couldn't process your request right now."

    def _generate_local(
        self,
        prompt: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate response using local Ollama model"""
        try:
            start_time = time.time()

            # Build full prompt with history
            full_prompt = self.system_prompt + "\n\n"
            for msg in conversation_history:
                role = msg["role"].capitalize()
                content = msg["content"]
                full_prompt += f"{role}: {content}\n"
            full_prompt += f"User: {prompt}\nAssistant:"

            # Call Ollama API
            response = requests.post(
                f"{self.local_host}/api/generate",
                json={
                    "model": self.local_model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")

            result = response.json()
            response_text = result.get("response", "").strip()
            generation_time = time.time() - start_time

            self.logger.info(
                f"Local response generated in {generation_time:.2f}s "
                f"({len(response_text)} chars)"
            )
            self.logger.debug(f"Response: '{response_text}'")

            return response_text

        except Exception as e:
            self.logger.error(f"Local generation failed: {e}")
            return "I'm having trouble processing your request right now."

    def get_model_info(self) -> Dict:
        """Get information about current model configuration"""
        return {
            "mode": self.mode,
            "cloud_provider": self.cloud_provider,
            "cloud_model": self.cloud_model,
            "local_model": self.local_model,
            "cloud_available": self.cloud_client is not None,
            "local_available": self._test_ollama_connection()
        }


if __name__ == "__main__":
    # Test the LLM Client
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test with local mode (no API key required)
    llm = LLMClient(mode="local")

    logger.info("Model info:")
    info = llm.get_model_info()
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Test generation
    prompt = "Tell me a very short joke"
    logger.info(f"\nPrompt: {prompt}")
    response = llm.generate_response(prompt)
    logger.info(f"Response: {response}")
