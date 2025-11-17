from typing import List

import requests
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
#  БАЗОВЫЕ МОДЕЛИ ЗАПРОСА
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "local"
    messages: List[Message]
    max_tokens: int = 128
    temperature: float = 0.7


# ============================================================
#  НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# ============================================================

app = FastAPI()

LLAMA_URL = "http://127.0.0.1:8000/completion"


# ============================================================
#  HEALTHCHECK
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
#  ОСНОВНОЙ ЧАТ-ЭНДПОИНТ
# ============================================================

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """
    Принимает OpenAI-совместимый запрос и прокидывает его в llama.cpp /completion.
    Формирует более простой, инструкционный промпт под Vikhr-7B.
    """

    # 1. Собираем system-контекст и последний user-вопрос
    system_parts = [m.content for m in req.messages if m.role.lower() == "system"]
    user_parts = [m.content for m in req.messages if m.role.lower() == "user"]

    system_text = "\n".join(system_parts).strip()
    if not system_text:
        system_text = "Ты умный офлайн-ассистент. Отвечай кратко и по делу."

    if user_parts:
        user_text = user_parts[-1].strip()
    else:
        user_text = ""

    # Итоговый промпт под инструкционную модель
    prompt = (
        f"{system_text}\n\n"
        f"Пользователь: {user_text}\n"
        f"Ассистент:"
    )

    n_predict = int(max(1, min(req.max_tokens, 256)))

    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        # при необходимости позже можно добавить temperature
        # "temperature": float(req.temperature),
    }

    print("=== PROXY → LLAMA REQUEST ===")
    print(payload)

    # 2. HTTP-запрос к llama.cpp через requests (как в debug_llama.py)
    try:
        r = requests.post(
            LLAMA_URL,
            json=payload,
            timeout=60,
        )
    except Exception as e:
        print("HTTP ERROR TO LLAMA:", repr(e))
        return {
            "error": "llama_http_error",
            "detail": repr(e),
        }

    status = r.status_code
    text = r.text

    print("=== LLAMA /completion RESPONSE ===")
    print("STATUS:", status)
    print("RAW   :", text[:500])

    if status != 200:
        return {
            "error": "llama_bad_status",
            "status_code": status,
            "text": text,
        }

        # 3. Парсим JSON
    try:
        data = r.json()
    except Exception as e:
        print("JSON PARSE ERROR:", repr(e))
        return {
            "error": "llama_not_json",
            "detail": repr(e),
            "raw_text": text,
        }

    # 4. Достаём текст ответа

    # исходный промпт, который мы отправляли
    original_prompt = payload.get("prompt", "")

    content = ""

    if isinstance(data, dict):
        # 4.1. сначала пытаемся взять data["content"] (вдруг модель его заполняет)
        content = data.get("content") or ""

        # 4.2. если content пустой, достаём ответ из поля "prompt"
        if not content:
            full_prompt = data.get("prompt") or ""
            if isinstance(full_prompt, str) and full_prompt:
                if original_prompt and full_prompt.startswith(original_prompt):
                    # хвост после исходного промпта = сгенерированный ответ
                    content = full_prompt[len(original_prompt):].strip()
                else:
                    content = full_prompt.strip()

    # если всё ещё пусто — последний fallback, чтоб не возвращать null
    if not content:
        content = str(data)

    # немного чистим начало — убираем возможные префиксы
    for prefix in ("Ассистент:", "Assistant:", "assistant:", "ASSISTANT:"):
        if content.startswith(prefix):
            content = content[len(prefix):].lstrip()
            break

    # 5. Отдаём OpenAI-совместимый ответ
    return {
        "id": "local-chat",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ]
    }
