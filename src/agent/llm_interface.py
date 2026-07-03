from __future__ import annotations

from http import HTTPStatus
from typing import Any, Optional

import dashscope
from dashscope import MultiModalConversation

from ..utils.path import get_config


config = get_config("agent.yaml")
api_key: str = config.get("openai_api_key", "")
base_url: str = str(config.get("openai_base_url", "")).rstrip("/")
model_name: str = config.get("llm_model_name", "")


def _dashscope_base_url() -> str:
    if base_url.endswith("/compatible-mode/v1"):
        return base_url[: -len("/compatible-mode/v1")] + "/api/v1"
    return base_url


class Message:
    """Represents a message for interacting with the LLM."""

    def __init__(self, role: str, content: Optional[str] = None):
        self.role = role
        self.content: list[dict[str, Any]] = []
        if content:
            self.add_text(content)

    def add_text(self, text: str):
        self.content.append({
            "type": "text",
            "text": text,
        })
        return self

    def add_image(self, image_path: str):
        self.content.append({
            "type": "image_url",
            "image_url": {"url": image_path},
        })
        return self

    def add_image_base64(self, image_base64: str):
        self.content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        })
        return self

    def add_video(self, video_path: str, fps: int = 2):
        self.content.append({
            "type": "video_url",
            "video_url": {"url": video_path},
            "fps": fps,
        })
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


def _content_to_dashscope(content: Any) -> list[dict[str, Any]] | str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    converted: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            converted.append({"text": str(item)})
            continue

        item_type = item.get("type")
        if item_type == "text":
            converted.append({"text": str(item.get("text", ""))})
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            converted.append({"image": image_url.get("url")})
        elif item_type == "video_url":
            video_url = item.get("video_url") or {}
            payload: dict[str, Any] = {"video": video_url.get("url")}
            if item.get("fps") is not None:
                payload["fps"] = item.get("fps")
            if item.get("max_frames") is not None:
                payload["max_frames"] = item.get("max_frames")
            converted.append(payload)
        elif "text" in item or "image" in item or "video" in item:
            converted.append(item)
        else:
            converted.append({"text": str(item)})
    return converted


def _messages_to_dashscope(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.get("role", "user"),
            "content": _content_to_dashscope(message.get("content", "")),
        }
        for message in messages
    ]


def _extract_text(response: Any) -> str:
    content = response.output.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("text")]
        return "\n".join(texts).strip()
    return str(content).strip()


def chat_with_llm(messages: list[dict[str, Any]]) -> str:
    """Chat with the LLM using DashScope native multimodal upload support."""
    dashscope.base_http_api_url = _dashscope_base_url()
    response = MultiModalConversation.call(
        api_key=api_key,
        model=model_name,
        messages=_messages_to_dashscope(messages),
    )
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope {response.status_code} {response.code}: {response.message}")
    return _extract_text(response)
