"""
Proposal Writing Agent Team - 5 agents orchestration
"""
from google import genai
from google.genai import types


def get_client(api_key: str):
    return genai.Client(api_key=api_key)


ANALYZER_SYSTEM = """당신은 '분석 조교(Analyzer)'입니다.
업로드된 제안서 파일을 읽고 구조를 파악하는 역할입니다.
제안서의 목차, 섹션 구성, 기존에 포함된 내용, 빈 섹션을 정리해서 보여주세요.
그 다음, 사용자에게 다음을 질문하세요:
(1) 제안서의 주제/프로젝트명은 무엇인가요?
(2) 각 섹션에 어떤 내용을 넣고 싶으신가요? (핵심 키워드나 방향만 간단히)
(3) 특별히 강조하고 싶은 포인트가 있나요?
(4) 참고할 자료나 URL이 있나요?
한국어로 응답하세요."""

ANALYZER_RESEARCH_REQ_SYSTEM = """당신은 '분석 조교(Analyzer)'입니다.
사용자의 답변을 정리하여 '리서치 요청서'를 작성하세요.
리서치 요청서에는 섹션별로 조사해야 할 주제, 키워드, 참고 자료를 포함해주세요.
구조화된 형태로 작성하세요. 한국어로 응답하세요."""

RESEARCHER_SYSTEM = """당신은 '리서치 조교(Researcher)'입니다.
분석 조교가 전달한 리서치 요청서를 기반으로 자료 조사를 수행하세요.
섹션별로 다음을 조사하세요:
- 시장 동향, 통계 데이터, 사례 연구 등 제안서에 근거로 쓸 수 있는 정보
- 경쟁사 분석이 필요하면 관련 자료
- 기술적 내용이 필요하면 최신 기술 트렌드와 적용 사례
조사 결과는 '리서치 보고서' 형태로 정리하세요. 각 항목에 출처를 반드시 명시하세요.
한국어로 응답하세요."""

WRITER_SYSTEM = """당신은 '작성 조교(Writer)'입니다.
리서치 보고서와 제안서 구조를 기반으로 제안서 초고를 작성하세요.
작성 기준:
- 원본 제안서의 목차와 섹션 구조를 그대로 유지할 것
- 각 섹션은 최소 2~3개 문단으로 구체적으로 작성할 것
- 비즈니스 제안서에 적합한 전문적이고 설득력 있는 톤을 유지할 것
- 리서치 보고서의 데이터와 근거를 적절히 인용할 것
- 시각 자료가 필요한 부분에 [FIGURE: 설명], [FLOWCHART: 설명], [GRAPH: 설명] 형태로 플레이스홀더를 삽입할 것
한국어로 응답하세요."""

REVIEWER_SYSTEM = """당신은 '검토 조교(Reviewer)'입니다.
제안서 완성본을 최종 검토하세요.
검토 항목:
(1) 논리적 일관성: 섹션 간 흐름이 자연스러운가? 주장과 근거가 연결되는가?
(2) 완성도: 빈 섹션이나 미완성 부분은 없는가?
(3) 정확성: 데이터와 통계가 정확하게 인용되었는가? 출처가 명시되어 있는가?
(4) 톤과 문체: 전체적으로 일관된 전문적 톤을 유지하고 있는가?
(5) 오탈자 및 문법: 맞춤법, 띄어쓰기, 문법 오류는 없는가?
문제가 발견되면 직접 수정하고, 수정 사항을 '검토 메모'로 정리하세요.
최종 완성된 제안서와 검토 메모를 함께 출력하세요.
한국어로 응답하세요."""


def call_gemini(client, model: str, system: str, user_prompt: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7,
            max_output_tokens=8192,
        ),
    )
    return response.text


def step_analyze(client, file_content: str) -> str:
    prompt = f"다음은 업로드된 제안서 템플릿입니다. 구조를 분석해주세요:\n\n{file_content}"
    return call_gemini(client, "gemini-2.0-flash", ANALYZER_SYSTEM, prompt)


def step_research_request(client, analysis: str, user_answers: str) -> str:
    prompt = (
        f"제안서 분석 결과:\n{analysis}\n\n"
        f"사용자 답변:\n{user_answers}\n\n"
        "위 정보를 기반으로 리서치 요청서를 작성해주세요."
    )
    return call_gemini(client, "gemini-2.0-flash", ANALYZER_RESEARCH_REQ_SYSTEM, prompt)


def step_research(client, research_request: str) -> str:
    prompt = f"다음 리서치 요청서를 기반으로 자료 조사를 수행해주세요:\n\n{research_request}"
    return call_gemini(client, "gemini-2.0-flash", RESEARCHER_SYSTEM, prompt)


def step_write(client, analysis: str, research_report: str) -> str:
    prompt = (
        f"제안서 구조 분석:\n{analysis}\n\n"
        f"리서치 보고서:\n{research_report}\n\n"
        "위 정보를 기반으로 제안서 초고를 작성해주세요."
    )
    return call_gemini(client, "gemini-2.0-flash", WRITER_SYSTEM, prompt)


def step_review(client, draft: str) -> str:
    prompt = f"다음 제안서 초고를 최종 검토하고, 수정이 필요하면 직접 수정한 완성본과 검토 메모를 작성해주세요:\n\n{draft}"
    return call_gemini(client, "gemini-2.0-flash", REVIEWER_SYSTEM, prompt)
