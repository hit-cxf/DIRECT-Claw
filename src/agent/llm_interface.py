from ..utils.path import get_config
from typing import Optional
from openai import OpenAI

config = get_config("agent.yaml")

client = OpenAI(
    api_key=config.get("openai_api_key"),
    base_url=config.get("openai_base_url")
)

model_name: str = config.get("llm_model_name", "")

class Message:
    """Represents a message for interacting with the LLM.

    Attributes:
        role (str): The role of the message (e.g., "user", "assistant", "system").
        content (str): The content of the message.
    """

    def __init__(self, role: str, content: Optional[str] = None):
        self.role = role
        self.content: list[dict] = []
        if content:
            self.add_text(content)
    def add_text(self, text: str):
        """Add text content to the message.

        Args:
            text (str): The text content.
        """
        self.content.append({
            "type": "text",
            "text": text
        })
        return self
    def add_image(self, image_path: str):
        """Add image content to the message.

        Args:
            image_path (str): The image file path.
        """
        self.content.append({
            "type": "image_url",
            "image_url": {"url": image_path},}
        )
        return self
    def add_image_base64(self, image_base64: str):
        """Add Base64 encoded image content to the message.

        Args:
            image_base64 (str): The Base64 encoded string of the image.
        """
        self.content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},}
        )
        return self
    def add_video(self, video_path: str, fps: int=2):
        """Add video content to the message.

        Args:
            video_path (str): The video file path.
        """
        self.content.append({
            "type": "video_url",
            "video_url": {"url": video_path},
            "fps": fps
        })
        return self
    
    def to_dict(self) -> dict:
        """Convert the message to a dictionary format.

        Returns:
            dict: A dictionary containing the role and content.
        """
        return {"role": self.role, "content": self.content}

def chat_with_llm(messages) -> str:
    """Chat with the LLM.

    Args:
        messages (list): The list of messages.
        verbose (bool): Whether to print detailed information in real-time.
    Returns:
        str: The response content from the LLM.
    """
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    result = chat_response.choices[0].message
    # if hasattr(result, "reasoning_content"):
    #     print(f"LLM reasoning content: {result.reasoning_content}")
    return result.content.strip() if result.content else ""
