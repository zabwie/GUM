"""Translate between OpenAI function calls and the target GGUF's Qwen tool format."""

import json
import re
import uuid


TOOL_START = "<tool_call>"
FUNCTION = re.compile(r"<function=([\w.-]+)>\s*\n(.*?)</function>", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>[ \t]*\r?\n(.*?)\r?\n</parameter>", re.DOTALL)


def normalize_messages(messages: list[dict]) -> list[dict]:
    normalized = []
    for message in messages:
        if message.get("role") == "developer":
            message = {**message, "role": "system"}
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Function call arguments must be a JSON object.")
                calls.append({"function": {**function, "arguments": arguments}})
            message = {**message, "tool_calls": calls}
        if message.get("role") == "system" and normalized and normalized[-1]["role"] == "system":
            earlier = normalized.pop()
            content = []
            for value in (earlier.get("content"), message.get("content")):
                if content:
                    content.append({"text": "\n\n"})
                content.extend(value if isinstance(value, list) else [{"text": value or ""}])
            message = {**message, "content": content}
        normalized.append(message)
    return normalized


def parse_reply(text: str, functions: dict[str, dict]) -> tuple[str, list[dict]]:
    """Extract complete calls for advertised tools; preserve malformed markup as text."""
    if not functions:
        return text, []

    calls = []

    def replace(block: re.Match) -> str:
        function = FUNCTION.fullmatch(block[1].strip())
        if not function or function[1] not in functions:
            return block[0]
        name, body = function.groups()
        schema = functions[name]
        if not isinstance(schema, dict):
            return block[0]
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        arguments = {}
        position = 0
        for parameter in PARAMETER.finditer(body):
            key, value = parameter.groups()
            if body[position:parameter.start()].strip() or key in arguments:
                return block[0]
            if schema.get("additionalProperties") is False and key not in properties:
                return block[0]
            definition = properties.get(key)
            param_type = definition.get("type") if isinstance(definition, dict) else None
            if param_type and param_type != "string":
                try:
                    value = json.loads(value)
                except ValueError:
                    if not (isinstance(param_type, list) and "string" in param_type):
                        return block[0]
            arguments[key] = value
            position = parameter.end()
        if body[position:].strip() or not set(schema.get("required", ())) <= arguments.keys():
            return block[0]
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
        return ""

    visible = re.sub(r"<tool_call>(.*?)</tool_call>", replace, text, flags=re.DOTALL)
    return visible, calls
