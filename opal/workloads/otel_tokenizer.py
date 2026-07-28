# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import json
import logging

from datetime import datetime, timezone

log = logging.getLogger("OtelTokenizer")


class OtelTokenizer:
    """Loads a HuggingFace tokenizer and converts OTel span message lists into token ID sequences."""

    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer

        log.info(f"Loading tokenizer '{tokenizer_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @staticmethod
    def _tool_result_text(content) -> str:
        if isinstance(content, list):
            return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        return str(content) if content is not None else ""

    @staticmethod
    def _system_instructions_to_messages(raw) -> list[dict]:
        """Parse gen_ai.system_instructions into a leading system message.

        The attribute is either a JSON string or an already-decoded list of
        parts (e.g. [{"type": "text", "content": "..."}]). Returns [] when
        absent/empty so it can be safely prepended to any conversation.
        """
        if not raw:
            return []
        parts = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parts, list):
            text = "\n".join(
                p.get("content", p.get("text", "")) for p in parts if isinstance(p, dict)
            )
        else:
            text = str(parts)
        if not text:
            return []
        return [{"role": "system", "parts": [{"type": "text", "content": text}]}]

    @staticmethod
    def _messages_to_chat_format(messages: list[dict]) -> list[dict]:
        """Convert OTel span messages (parts-based or content-based) to HuggingFace chat format."""
        result = []
        for msg in messages:
            # Already in standard {role, content} format
            if "content" in msg and "parts" not in msg:
                result.append({"role": msg.get("role", "user"), "content": str(msg["content"])})
                continue

            role = msg.get("role", "user")
            text_parts: list[str] = []
            tool_calls: list[dict] = []

            for part in msg.get("parts", []):
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("content", ""))
                elif ptype == "thinking":
                    # Reasoning tokens the model actually emitted -- keep them so they
                    # are tokenized/counted. The payload lives under "thinking".
                    text_parts.append(part.get("thinking") or part.get("content", ""))
                elif ptype == "tool_call":
                    tool_calls.append({
                        "id": part.get("id") or part.get("name") or "tool_call",
                        "type": "function",
                        "function": {
                            "name": part.get("name", ""),
                            "arguments": json.dumps(part.get("arguments", {})),
                        },
                    })
                elif ptype in ("tool_result", "tool_call_response"):
                    # Tool output fed back to the model. deepseek/agentic traces put
                    # the payload under "result"; others under "content".
                    payload = part.get("content", part.get("result", ""))
                    result.append({
                        "role": "tool",
                        "tool_call_id": part.get("id") or part.get("name") or "tool_call",
                        "content": OtelTokenizer._tool_result_text(payload),
                    })

            if tool_calls:
                # Llama-3.1's chat template only renders message['content'] in its
                # plain-text branch -- the moment a message has 'tool_calls', the
                # template's tool-call branch never looks at 'content' at all, so any
                # prose there would be silently dropped. Emit it as its own preceding
                # text-only message instead, so it round-trips through the template.
                if text_parts:
                    result.append({"role": role, "content": "\n".join(text_parts)})
                # The template also rejects more than one tool call per message
                # ("This model only supports single tool-calls at once!") -- split
                # multiple tool_call parts into one message each defensively.
                for call in tool_calls:
                    result.append({"role": role, "content": None, "tool_calls": [call]})
            elif text_parts:
                result.append({"role": role, "content": "\n".join(text_parts)})

        return result

    @staticmethod
    def _tools_to_chat_format(tool_defs: list[dict]) -> list[dict]:
        """Convert gen_ai.tool.definitions into the apply_chat_template tools format."""
        result = []
        for t in tool_defs:
            fn = t.get("function", t)
            result.append({
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", fn.get("input_schema", {})),
                },
            })
        return result

    @staticmethod
    def _to_ids(result) -> list[int]:
        """Normalise apply_chat_template output to a plain list[int].

        Some transformers versions return a BatchEncoding (dict-like) rather than
        a bare list when tokenize=True, especially when tools= is passed.
        """
        if isinstance(result, list):
            return result
        # BatchEncoding / dict-like: extract input_ids
        ids = result.get("input_ids") if hasattr(result, "get") else getattr(result, "input_ids", None)
        if ids is not None:
            return ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return list(result)

    def _template_supports_tools(self) -> bool:
        """Return True if the tokenizer's chat template natively injects tool definitions."""
        if not self.tokenizer.chat_template:
            return False
        probe_msg = [{"role": "user", "content": "hi"}]
        probe_tool = [{"type": "function", "function": {"name": "_probe", "description": "", "parameters": {}}}]
        try:
            with_tools = self.tokenizer.apply_chat_template(probe_msg, tools=probe_tool, tokenize=False, add_generation_prompt=False)
            without_tools = self.tokenizer.apply_chat_template(probe_msg, tokenize=False, add_generation_prompt=False)
            return "_probe" in with_tools and with_tools != without_tools
        except Exception:
            return False

    def tokenize_messages(
        self,
        messages: list[dict],
        tool_defs: "list[dict] | None" = None,
        system_instructions=None,
    ) -> list[int]:
        """Tokenize input messages using the model's chat template.

        If the tokenizer does not natively support the tools= kwarg (e.g. Llama-3-8B),
        tool definitions are serialized as a JSON system message prepended to the
        conversation so that token counts remain accurate for tool-heavy traces.

        gen_ai.system_instructions, when present, is prepended as a leading system
        message so the system prompt is counted just as the provider sent it.
        """
        import json as _json
        sys_msgs = self._system_instructions_to_messages(system_instructions)
        chat_messages = self._messages_to_chat_format(sys_msgs + messages)
        chat_tools = self._tools_to_chat_format(tool_defs) if tool_defs else None
        kwargs: dict = dict(conversation=chat_messages, tokenize=True, add_generation_prompt=True)

        if chat_tools:
            if self._template_supports_tools():
                kwargs["tools"] = chat_tools
            else:
                # Tokenizer ignores tools= silently — prepend them as a system message
                tool_json = _json.dumps(chat_tools, separators=(",", ":"))
                system_msg = {"role": "system", "content": f"Tools available:\n{tool_json}"}
                kwargs["conversation"] = [system_msg] + chat_messages

        try:
            return self._to_ids(self.tokenizer.apply_chat_template(**kwargs))
        except Exception as e:
            log.warning(f"apply_chat_template failed ({e}), falling back to plain concatenation")
            parts = [f"{m['role']}: {m.get('content') or ''}" for m in chat_messages]
            return self.tokenizer.encode("\n".join(parts))

    def tokenize_output(
        self,
        input_messages: list[dict],
        output_messages: list[dict],
        system_instructions=None,
    ) -> list[int]:
        """Return only the new tokens the model emitted (assistant turn in context).

        Computes full_conversation_tokens - input_tokens so that the result can be
        appended to hash_ids without introducing a spurious BOS token that would
        break prefix matching for subsequent turns.

        Output tokens are appended to hash_ids after request completion so that
        the full context (prompt + response) is stored in the KV cache, enabling
        prefix hits for subsequent turns that include this response in their history.

        The same system_instructions passed to tokenize_messages must be passed here
        so the input side of the diff matches and only the assistant tokens remain.
        """
        chat_output = self._messages_to_chat_format(output_messages)
        if not chat_output:
            return []
        sys_msgs = self._system_instructions_to_messages(system_instructions)
        chat_input = self._messages_to_chat_format(sys_msgs + input_messages)
        full_conversation = chat_input + chat_output
        try:
            input_ids = self._to_ids(self.tokenizer.apply_chat_template(
                chat_input, tokenize=True, add_generation_prompt=True
            ))
            full_ids = self._to_ids(self.tokenizer.apply_chat_template(
                full_conversation, tokenize=True, add_generation_prompt=False
            ))
            output_ids = full_ids[len(input_ids):]
            return output_ids
        except Exception as e:
            log.warning(f"apply_chat_template failed for output ({e}), falling back to plain encode")
            parts = [m.get("content") or "" for m in chat_output]
            text = " ".join(p for p in parts if p)
            return self.tokenizer.encode(text, add_special_tokens=False) if text else []

    @staticmethod
    def parse_iso(ts: str) -> datetime:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def is_multi_session(trace_file: str) -> bool:
        """Return True if the file contains multiple sessions (JSONL or a JSON list of sessions)."""
        if trace_file.endswith(".jsonl"):
            return True
        with open(trace_file, "r") as f:
            for ch in f.read(256):
                if ch in "[\n\r\t ":
                    if ch == "[":
                        return True
                    continue
                return False
        return False

    @staticmethod
    def load_spans(trace_file: str) -> tuple[list[dict], datetime]:
        """Load and sort OTel spans from a JSON file. Returns (spans, base_time)."""
        import json

        with open(trace_file, "r") as f:
            data = json.load(f)

        total_in_file = len(data.get("spans", []))
        spans = [
            s for s in data["spans"]
            if "gen_ai.input.messages" in s.get("attributes", {})
        ]
        if not spans:
            raise ValueError(f"No spans with gen_ai.input.messages found in {trace_file}")

        spans.sort(key=lambda s: s["start_time"])
        base_time = OtelTokenizer.parse_iso(spans[0]["start_time"])

 #      log.debug(
 #           f"[load_spans] {trace_file}: {total_in_file} total spans, {len(spans)} with gen_ai.input.messages"
 #           f" | base_time={base_time.isoformat()}"
 #           f" | first span_id={spans[0].get('span_id','?')} trace_id={spans[0].get('trace_id','?')}"
 #           f" | last  span_id={spans[-1].get('span_id','?')} end_time={spans[-1].get('end_time','?')}"
 #           f" | wall_duration={(OtelTokenizer.parse_iso(spans[-1]['start_time']) - base_time).total_seconds():.3f}s"
 #       )
        return spans, base_time

    @staticmethod
    def load_sessions(trace_file: str) -> list[tuple[list[dict], datetime]]:
        """Load a multi-session trace file (JSONL or JSON list) as a list of (spans, base_time) per session."""
        import json

        def _parse_session(obj):
            spans = [
                s for s in obj.get("spans", [])
                if "gen_ai.input.messages" in s.get("attributes", {})
            ]
            if not spans:
                return None
            spans.sort(key=lambda s: s["start_time"])
            base_time = OtelTokenizer.parse_iso(spans[0]["start_time"])
            return (spans, base_time)

        sessions = []
        # add check trace_file exists and is not empty
        if not os.path.isfile(trace_file) or os.path.getsize(trace_file) == 0:
            raise ValueError(f"Trace file {trace_file} does not exist or is empty")

        if trace_file.endswith(".jsonl"):
            with open(trace_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    result = _parse_session(json.loads(line))
                    if result:
                        sessions.append(result)
        else:
            with open(trace_file, "r") as f:
                data = json.load(f)
            # a single session saved as one JSON object (not wrapped in a list)
            if isinstance(data, dict):
                data = [data]
            for obj in data:
                result = _parse_session(obj)
                if result:
                    sessions.append(result)

        if not sessions:
            raise ValueError(f"No sessions with gen_ai.input.messages found in {trace_file}")

        total_spans = sum(len(s) for s, _ in sessions)
#        log.debug(
#            f"[load_sessions] {trace_file}: {len(sessions)} sessions, {total_spans} total spans"
#        )
        for i, (spans, base_time) in enumerate(sessions):
            trace_id = spans[0].get("trace_id", "?")
            wall_end = spans[-1].get("end_time", "?")
#            log.debug(
#                f"[load_sessions]  session {i + 1}/{len(sessions)}"
#                f" trace_id={trace_id} spans={len(spans)}"
#                f" base_time={base_time.isoformat()} last_end={wall_end}"
#            )
        return sessions
