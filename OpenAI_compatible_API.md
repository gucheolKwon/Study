안녕하세요. 제공해주신 코드를 분석해보니 OpenWeb-UI와 정상적으로 연동되지 않는 몇 가지 중요한 이유가 있습니다. 주로 OpenAI API 사양과의 미묘한 불일치 및 데이터 처리 방식에 문제가 있습니다.

아래에 문제점들을 상세히 분석하고, 수정된 전체 코드를 제공해 드리겠습니다.

주요 문제점 분석
1. 대화 내역(Conversation History) 처리 오류 (가장 큰 문제)
OpenWeb-UI는 이전 대화 내용을 포함한 전체 메시지 목록을 messages 배열에 담아 보냅니다. 하지만 현재 코드는 이 목록에서 첫 번째 메시지만 사용하고 있습니다.

# 문제의 코드
user_question = data.get("messages")[0].get("content")
payload = json.dumps({
    # ...
    "messages": [
        {"role": "system", "content": context_2},
        {"role": "user", "content": f"{str(user_question)}"} #  MISTAKE: 항상 첫 번째 메시지만 사용
    ],
    # ...
})
content_copy
download
Use code with caution.
Python
이렇게 하면 다음과 같은 문제가 발생합니다.

문맥 유지 불가능: 사용자가 "이전 답변에 대해 더 설명해줘"라고 질문하면, API는 이전 대화 내용을 전혀 모르기 때문에 엉뚱한 답변을 하게 됩니다.
잘못된 질문 전달: 대화가 길어지면 messages 배열에는 user, assistant 역할이 번갈아 나타납니다. messages[0]은 대화의 가장 첫 질문일 뿐, 현재 질문이 아닐 수 있습니다. 항상 마지막 메시지가 현재 사용자의 질문입니다.
해결책: OpenWeb-UI가 보낸 messages 목록 전체를 그대로 또는 약간 수정하여 백엔드 LLM에 전달해야 합니다.

2. 모델 선택 무시
OpenWeb-UI에서 사용자가 특정 모델(예: Gauss2-37b-instruct-v1.0)을 선택하면, 이 모델 이름이 요청 body에 포함되어 전달됩니다. 하지만 현재 코드는 이 값을 무시하고 항상 "custom_llm_model"로 하드코딩된 모델을 호출합니다.

# 문제의 코드
payload = json.dumps({
    "model": "custom_llm_model", # MISTAKE: 사용자가 선택한 모델을 무시함
    # ...
})
content_copy
download
Use code with caution.
Python
이는 유연성을 떨어뜨리고, 사용자가 UI에서 모델을 변경해도 실제로는 아무런 변화가 없는 문제를 일으킵니다.

해결책: 요청에서 받은 model 값을 백엔드 LLM에 전달할 payload에 사용해야 합니다.

3. "가짜 스트리밍"으로 인한 초기 응답 지연
현재 스트리밍 방식은 백엔드 LLM으로부터 모든 텍스트를 한 번에 받은 후, FastAPI단에서 이것을 잘게 쪼개어 스트리밍처럼 보이게 만듭니다.

# 현재 방식
1. OpenWeb-UI -> FastAPI (요청)
2. FastAPI -> Custom LLM (요청)
3. (대기...) Custom LLM이 모든 문장을 생성할 때까지 기다림
4. Custom LLM -> FastAPI (전체 응답)
5. FastAPI가 전체 응답을 잘게 쪼개서 -> OpenWeb-UI (스트리밍 시작)
content_copy
download
Use code with caution.
Python
이 방식의 가장 큰 문제는 3번 단계에서 긴 지연이 발생한다는 점입니다. LLM이 1분 동안 답변을 생성한다면, 사용자는 1분 동안 아무런 피드백 없이 기다려야 합니다. OpenWeb-UI는 타임아웃으로 연결을 끊거나 "No response from server" 오류를 표시할 수 있습니다.

근본적인 해결책: 백엔드 custom_llm_endpoint가 스트리밍 응답(text/event-stream)을 지원하도록 변경하고, FastAPI는 그 스트림을 그대로 중계(proxy)하는 것이 가장 좋습니다.

임시 해결책 (현재 코드 유지 시): 백엔드 LLM이 스트리밍을 지원하지 않는다면 현재 방식이 유일한 대안입니다. 다만, 이로 인해 타임아웃이 발생할 수 있다는 점을 인지해야 합니다. 코드는 이 구조를 유지하되, 다른 문제들을 수정하여 최소한 작동은 되도록 만들어야 합니다.

4. FastAPI의 Pydantic 모델 미사용
Request 객체를 직접 파싱하는 대신, 처음에 정의한 ChatRequest Pydantic 모델을 FastAPI의 의존성 주입 기능으로 활용하는 것이 더 좋습니다. 이렇게 하면 코드 가독성이 향상되고, 들어오는 요청에 대한 유효성 검사를 자동으로 수행해 줍니다.

# 현재 방식
async def get_answer(question: Request):
    data = await question.json()
    model = data.get("model")
    # ...

# 권장 방식
async def get_answer(chat_request: ChatRequest):
    model = chat_request.model
    # ...
content_copy
download
Use code with caution.
Python
수정된 전체 코드
위 문제점들을 모두 수정한 코드입니다. 이 코드를 사용하면 OpenWeb-UI와 정상적으로 연동될 것입니다.

from fastapi import FastAPI, HTTPException, Request
import json
import httpx
import uuid
import os
import re
import time
import asyncio
from pprint import pprint
from typing import List, Dict, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# --- Pydantic 모델 정의 (변경 없음) ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.4
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None

    class Config:
        extra = "allow"

# --- FastAPI 앱 및 미들웨어 설정 (변경 없음) ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 스트리밍 응답 생성기 (개선) ---
async def stream_response_generator(response_data: dict) -> AsyncGenerator[str, None]:
    """
    백엔드에서 받은 전체 응답을 OpenAI 스트리밍 형식(SSE)으로 변환하여 생성합니다.
    """
    response_id = f"chatcmpl-{uuid.uuid4()}"
    model_name = response_data.get("model", "custom-model")
    created_time = int(time.time())
    
    # 1. 시작 청크 (역할 정보)
    start_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(start_chunk)}\n\n"
    
    # 2. 콘텐츠 청크 (실제 답변 내용)
    # 백엔드 API가 스트리밍을 지원하지 않으므로, 받은 전체 텍스트를 잘라서 보냅니다.
    full_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # 예시: 5자씩 잘라서 스트리밍 효과를 줌
    chunk_size = 5
    for i in range(0, len(full_content), chunk_size):
        chunk_text = full_content[i:i+chunk_size]
        content_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"
        await asyncio.sleep(0.02) # 스트리밍 효과를 위한 아주 짧은 지연

    # 3. 종료 청크
    end_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_chunk)}\n\n"
    
    # 4. SSE 종료 신호
    yield "data: [DONE]\n\n"

# --- 핵심 API 엔드포인트 (대폭 수정) ---
@app.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatRequest):
    # 💡 FIX 4: Pydantic 모델을 직접 사용하여 요청을 받고 유효성을 검사합니다.
    try:
        # ⚠️ 중요: 실제 환경에 맞게 수정해주세요.
        api_base_url = 'YOUR_CUSTOM_LLM_ENDPOINT'  # 예: http://127.0.0.1:8080/ask
        credential_key = 'YOUR_CREDENTIAL_KEY'
        
        system_prompt = "아는 것에 대해서만 대답해주세요."

        # 💡 FIX 1: 전체 대화 내역을 사용합니다.
        # 시스템 프롬프트를 메시지 목록의 시작 부분에 추가합니다.
        messages_for_payload = [Message(role="system", content=system_prompt).dict()]
        messages_for_payload.extend([msg.dict() for msg in chat_request.messages])

        payload = {
            # 💡 FIX 2: OpenWeb-UI에서 사용자가 선택한 모델을 사용합니다.
            "model": chat_request.model,
            "messages": messages_for_payload,
            "temperature": chat_request.temperature,
            "stream": False,  # 백엔드 API는 스트리밍을 지원하지 않으므로 항상 False
        }
        
        headers = {
            'x-dep-ticket': credential_key,
            'Send-System-Name': 'D0aKG',
            'User-Id': 'gucheol.kwon',
            'User-Type': 'gucheol.kwon',
            'Prompt-Msg-Id': str(uuid.uuid4()),
            'Completion-Msg-Id': str(uuid.uuid4()),
            'Content-Type': 'application/json'
        }
        
        # httpx 클라이언트를 사용하여 비동기 요청
        # 💡 FIX 3 (Latency): 타임아웃을 늘려서 백엔드 LLM이 응답을 생성할 시간을 줍니다.
        async with httpx.AsyncClient(timeout=300.0) as client: # 5분 타임아웃
            response = await client.post(
                url=api_base_url,
                headers=headers,
                json=payload  # json.dumps 대신 json 파라미터 사용 권장
            )
        
        # 응답 상태 코드 확인
        response.raise_for_status()
            
        # 백엔드에서 받은 응답
        backend_response_json = response.json()

        # 💡 OpenWeb-UI는 항상 스트리밍 응답을 기대하므로,
        # 백엔드 응답을 스트리밍 형식으로 변환하여 반환합니다.
        return StreamingResponse(
            stream_response_generator(backend_response_json),
            media_type="text/event-stream"
        )
        
    except httpx.HTTPStatusError as e:
        print(f"Error calling backend API: {e}")
        print(f"Response body: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 모델 목록 엔드포인트 (수정) ---
@app.get("/v1/models")
def list_models():
    # 여기에 OpenWeb-UI에서 선택 가능하게 할 모델 목록을 정의합니다.
    # "id" 값이 ChatRequest의 'model' 필드로 전달됩니다.
    return {
        "object": "list",
        "data": [
            # 💡 FIX 2 (연관): 이 id가 chat_completions에서 사용됩니다.
            # 백엔드 LLM이 실제로 지원하는 모델 이름을 id로 사용해야 합니다.
            {"id": "Gauss2-37b-instruct-v1.0", "object": "model", "owned_by": "custom", "permission": []},
            {"id": "another-custom-model-id", "object": "model", "owned_by": "custom", "permission": []}
        ]
    }
content_copy
download
Use code with caution.
Python
수정된 코드 사용 방법
코드 교체: 위의 전체 코드를 기존 파일에 붙여넣으세요.
엔드포인트 및 키 수정: chat_completions 함수 내의 api_base_url과 credential_key를 실제 값으로 반드시 변경해주세요.
모델 ID 확인: /v1/models 엔드포인트에 나열된 id 값(예: Gauss2-37b-instruct-v1.0)이 백엔드 LLM에서 실제로 인식하는 모델 이름과 일치하는지 확인하세요.
서버 실행: FastAPI 서버를 실행합니다.
OpenWeb-UI 설정:
Settings -> Connections -> API 로 이동합니다.
API Base URL에 FastAPI 서버 주소(예: http://localhost:8000/v1)를 입력합니다.
API Key는 아무 값이나 입력해도 됩니다 (현재 코드에서는 사용하지 않음).
모델 목록에서 Gauss2-37b-instruct-v1.0 등을 선택하고 대화를 시작합니다.
이제 OpenWeb-UI가 문맥을 올바르게 유지하고, 사용자가 선택한 모델로 요청을 보내며, 스트리밍처럼 보이는 응답을 정상적으로 수신할 수 있을 것입니다.
