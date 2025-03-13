LLM
- invoke : LLM 호출
- batch : 여러개의 인풋을 독립적으로 호출 (parallel processing)
- stream : Output streaming
- ainvoke / abatch / astream
- PromptTemplate : LLM 인풋을 위한 prompt 객체. from_template 메서드를 사용해서 구체적으로 정의
- ChatPromptTemplate : Prompt를 대화형으로 제공. from_message 메서드를 통해 role, message를 대화형으로 추가하여 순차적인 prompt 생성 가능
- Langchain Expression Language (LCEL) : LangChain의 컴포넌트들을 블럭처럼 연결. 이전 component의 ouptut을 다음의 인풋으로 전달하는 방식
- BaseChatMemory : 대화기록 가져와서 (과거 히스토리) 고려함.
- StrOutputParser : LLM 답변 중 content만 자동으로 추출하는 Tool


```python
from langchain_core.messages import HumanMessage, SystemMessage
# HumanMessage: 대화형 상황에서 인간 사용자가 입력한 메시지를 나타냄
# SystemMessage: 시스템 또는 모델에서 나온 메시지를 나타내며, 주로 지시사항이나 응답을 설정하는 역할을 함

from langchain_core.output_parsers import StrOutputParser
# StrOutputParser: 모델의 출력값을 받아서 문자열 형식으로 변환하는 파서
# 모델의 응답을 일관성 있게 처리하고 구조화하는 데 유용함

from langchain_core.prompts import ChatPromptTemplate
# ChatPromptTemplate: 모델에게 입력할 프롬프트(지시사항)를 어떻게 구성할지 정의하는 템플릿
# 대화를 위한 지침이나 콘텐츠를 템플릿 형식으로 설정할 수 있게 해줌
```
```python
messages = [
    SystemMessage("당신은 친절한 AI 어시스턴트 입니다."), #LLM에게 역할 부여
    HumanMessage("한글로 당신을 소개해주세요."), #LLM에게 보내는 프롬프트
]

parser = StrOutputParser()
# pipe (|) 연산자를 통해 두 객체를 연결해서 하나의 체인으로 만들 수 있습니다.
chain = llm | parser

from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

# 라이브러리 및 모듈 설명:
#   - List: 타입 힌트를 제공하여 리스트 형태의 데이터를 처리할 수 있도록 지원
#   - BaseModel: 데이터 검증 및 구조화를 위한 기본 클래스
#   - Field: 필드별 메타데이터를 추가하여 설명을 제공하고, 유효성 검사를 지원
#   - PydanticOutputParser: LLM의 출력을 Pydantic 데이터 모델로 변환하는 파서
#   - OutputFixingParser: LLM 출력이 유효한 형식이 아닐 경우 자동으로 수정하는 파서

# 음식 정보를 저장하는 데이터 모델 정의
class Food(BaseModel):
    name: str = Field(description="Name of a food")  # name이라는 필드의 형식은 문자형이고 음식 이름을 나타냅니다.
    ingredients: List[str] = Field(description="List of names of the ingredients mentioned")  # inrgredients라는 필드의 형식은 문자형 요소로 구성된 리스트이고 음식 재료 목록을 나타냅니다.
# PydanticOutputParser: LLM의 출력을 Food 모델에 맞게 변환하는 파서
parser = PydanticOutputParser(pydantic_object=Food)
# OutputFixingParser: LLM의 출력이 유효한 형식이 아닐 경우 LLM을 사용하여 자동 수정하는 파서
new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

# 메모리 추가하기
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
memory.save_context(
    inputs={
        "user": "내가 영화를 보고 싶은데 어떤걸 봐야 할지 모르겠어."
    },
    outputs={
        "assistant": "안녕하세요, 영화 추천이라고 하면 제가 전문이죠! 어떤 장르의 영화를 보고싶으신가요?"
    },
)

# LangChain에서 제공하는 대화 히스토리 관련 클래스 임포트
from langchain_core.chat_history import (
    BaseChatMessageHistory,  # 모든 대화 히스토리 클래스의 기본이 되는 추상 클래스
    InMemoryChatMessageHistory,  # 메모리 기반으로 대화 기록을 저장하는 클래스
)

# LangChain에서 대화 히스토리를 관리하면서 실행할 수 있는 래퍼(wrapper-감싸주는) 클래스 임포트
from langchain_core.runnables.history import RunnableWithMessageHistory
```

RAG
```python
# 'itemgetter'는 다수의 항목을 기준으로 정렬하거나 선택할 때 유용한 유틸리티 함수입니다.
# 특정 키 또는 인덱스를 기준으로 정렬하거나 값을 추출할 수 있습니다.
from operator import itemgetter

# ChatOllama는 Ollama라는 특정 모델을 사용하는 Langchain의 대화형 AI 모델 클래스입니다.
# Langchain의 커뮤니티에서 제공하는 'ChatOllama' 모델을 불러옵니다.
from langchain_community.chat_models import ChatOllama

# OllamaEmbeddings는 Ollama 모델로부터 임베딩을 생성하기 위한 클래스입니다.
# 텍스트를 숫자 벡터로 변환하여 임베딩을 생성할 수 있게 도와줍니다.
from langchain_community.embeddings import OllamaEmbeddings

# FAISS는 벡터 검색 엔진입니다. 임베딩된 데이터를 인덱싱하고 유사도를 빠르게 계산하여
# 필요한 데이터를 검색할 수 있도록 돕는 역할을 합니다.
from langchain_community.vectorstores import FAISS

# Document는 Langchain에서 문서를 처리할 때 사용하는 기본 데이터 구조입니다.
# 주로 텍스트와 해당 텍스트의 메타데이터를 함께 관리하는데 사용됩니다.
from langchain_core.documents import Document

# StrOutputParser는 LLM(Large Language Model)의 출력을 파싱할 때 문자열로 변환하는 데 사용됩니다.
# 대화나 텍스트 생성에서 모델의 출력을 후처리하는 도구입니다.
from langchain_core.output_parsers import StrOutputParser

# ChatPromptTemplate은 LLM에 대한 프롬프트(질문 또는 요청)를 설정하고 관리하는 클래스입니다.
# 사용자로부터 입력을 받아 구조화된 프롬프트를 생성하여 LLM에 전달할 수 있도록 합니다.
from langchain_core.prompts import ChatPromptTemplate

# OllamaEmbeddings 생성
embeddings = OllamaEmbeddings(model="mistral:7b")

documents = [
    Document(
        page_content="random.seed(a=None, version=2) 난수 생성기를 초기화합니다. a가 생략되거나 None이면, 현재 시스템 시간이 사용됩니다. 운영 체제에서 임의성 소스(randomness sources)를 제공하면, 시스템 시간 대신 사용됩니다 (가용성에 대한 자세한 내용은 os.urandom() 함수를 참조하십시오).",
        metadata={"source": "random.seed"},
    ),
    Document(
        page_content="math.gcd(*integers) 지정된 정수 인자의 최대 공약수를 반환합니다. 인자 중 하나가 0이 아니면, 반환된 값은 모든 인자를 나누는 가장 큰 양의 정수입니다. 모든 인자가 0이면, 반환 값은 0입니다. 인자가 없는 gcd()는 0을 반환합니다.",
        metadata={"source": "math.gcd"},
    ),
    Document(
        page_content="re.search(pattern, string, flags=0) string을 통해 스캔하여 정규식 pattern이 일치하는 첫 번째 위치를 찾고, 대응하는 일치 객체를 반환합니다. 문자열의 어느 위치도 패턴과 일치하지 않으면 None을 반환합니다; 이것은 문자열의 어떤 지점에서 길이가 0인 일치를 찾는 것과는 다르다는 것에 유의하십시오.",
        metadata={"source": "re.search"},
    ),
    Document(
        page_content="copy.deepcopy(x[, memo]) x의 깊은 사본을 반환합니다.",
        metadata={"source": "copy.deepcopy"},
    )
]

# FAISS(Facebook AI Similarity Search)는 대규모 벡터를 빠르게 검색하기 위한 라이브러리입니다.
# from_documents 메서드를 사용하여 주어진 문서와 임베딩을 기반으로 벡터 스토어(vector store)를 생성합니다.

vectorstore = FAISS.from_documents(  # FAISS 벡터 스토어 생성
    documents,                       # 문서 리스트 또는 텍스트 데이터, 각 문서는 벡터로 변환될 데이터입니다.
    embedding=embeddings,            # 임베딩 객체 또는 함수로, 문서의 텍스트를 벡터로 변환합니다.
)

# 결과적으로 FAISS는 문서의 임베딩 벡터를 저장하고, 유사도를 기반으로 빠른 검색을 수행할 수 있게 됩니다.

db_retriever = vectorstore.as_retriever()

db_retriever.invoke("파이썬에서 최대 공약수를 구하는 방법")

# 체인 구성
# itemgetter는 딕셔너리에서 특정 키의 값을 가져오는 함수를 생성합니다.
# 즉, 사용자가 입력한 role과 question에 더해
# context를 가져오는 체인을 활용해서 추출한 Document를 "context"에 넣어서 사용자에게 제공합니다.
qa_chain = (
    {"context": db_retriever | get_first_doc, "role": itemgetter("role"), "question": itemgetter("question")}
    | prompt_with_context
    | llm
    | StrOutputParser()
)
```
- text embedding : context가 중요하기 때문에 코사인 유사도를 씀
- image embedding : 전체 크기 차이를 고려하는게 중요하기 때문에 유클리디안(L2) 거리나 내적(dot product) 많이 사용
- io.StringIO() : 문자열을 메모리에 파일처럼 저장할 수 있는 객체를 생성합니다. 이후 표준 출력(print)으로 나오는 내용을 이 StringIO 객체에 저장합니다.

## LangGraph
- Class : 그래프에서 사용할 데이터 정의
- StateGraph : 그래프 생성
- Node : 특정 job 실행
- ToolNode : Agent가 사용할 수 있는 외부 tool 정의
- Edge : Node들을 연결하고, Agent의 작동 flow 정의
- 1) 클래스 생성하여 그래프에서 사용될 state의 형식을 구성. 2) 그래프에서 사용할 함수 구성 (def). 3) 그래프 생성 (Stategraph). 4) 노드와 엣지를 구성 (add_edge, add_node). 5) 컴파일 (compile)
```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
class AgentState(TypedDict): # 그래프에 사용될 데이터 형태가 딕셔너리라서 데이터의 key와 값의 타입을 정의합니다.

    messages: Annotated[str, add_messages] #상태를 업데이트 할 때 add_messages 메서드를 이용하여 messages에 내용 추가

#**1. class AgentState(TypedDict):**
#AgentState라는 클래스 생성
#**2. TypedDict**
#TypedDict는 딕셔너리와 비슷한 구조를 가진 클래스로,
#각 키와 값의 타입을 미리 정의할 수 있습니다.
#**3. messages: Annotated[str, add_messages]**
#messages의 데이터타입은 str이고 add_messages라는 메타 데이터를 갖습니다.
#메타 데이터
#'데이터에 대한 데이터'를 의미합니다.
#어떤 데이터에 대한 추가적인 정보를 제공해주는 데이터입니다.

loader = PyPDFLoader("Maximizing Muscle Hypertrophy.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


```
