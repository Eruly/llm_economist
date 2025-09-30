# LLM-Economist 제품 요구사항 문서 (PRD)

## 1. 목적과 배경
LLM-Economist는 대규모 언어모델(LLM)을 활용해 다양한 경제 주체(에이전트)로 구성된 경제 시뮬레이션을 수행하고, 조세정책 최적화 및 메커니즘 디자인을 탐구하기 위한 프레임워크입니다. 본 PRD는 프레임워크의 제품화 목표, 사용자 가치, 핵심 기능, 시스템 요구사항, 품질 기준, 릴리스 범위를 정의합니다.

## 2. 타깃 사용자 및 핵심 가치
- 연구자/데이터 사이언티스트: 경제 메커니즘, 조세정책, 대규모 다중 에이전트 상호작용을 재현·실험
- 정책 분석가: 합리/제한합리/민주적 의사결정 시나리오 하에서 정책 성과 비교
- 개발자/엔지니어: 다양한 LLM(클라우드/로컬)과 손쉽게 통합하여 대규모 실험 자동화

핵심 가치
- 다양한 LLM과 시나리오 지원으로 재현 가능한 실험 환경 제공
- 대규모 인구(3~1000+ 에이전트) 시뮬레이션을 효율적으로 실행
- 인구통계 기반 페르소나와 유틸리티 설계로 사실성/다양성 확보

## 3. 범위 (MVP 및 이후)
- MVP
  - 시뮬레이션 시나리오: `rational`, `bounded`, `democratic`, `fixed`
  - LLM 연동: OpenAI, OpenRouter, Google Gemini, vLLM, Ollama
  - 기본 UI: Streamlit 대시보드로 파라미터 설정/실행/로그 모니터링
  - 실험 스크립트: 논문 재현용 `experiments/run_experiments.py`
  - 기록/재시작: LLM 응답 히스토리와 메시지 기록(json/jsonl) 저장·로드·리플레이
- 확장(포스트-MVP)
  - 추가 프롬프트 전략(`sc`, `tot`, `mcts`) 안정화 및 하이퍼파라미터 탐색 지원
  - 에이전트 유틸리티 커스터마이징 템플릿, 정책학습 자동화
  - 분산 실행/큐 기반 배치 러너, 비용/성능 대시보드(비용 추적, 토큰 사용량)

## 4. 사용자 시나리오
- 빠른 시작: CLI 또는 Streamlit에서 시나리오/에이전트 수/LLM 모델 선택 → 20~1000 타임스텝 실행 → 결과 로그 및 베스트 타임스텝 조회
- 정책 비교: 동일 인구/유틸리티 하에 서로 다른 조세 브래킷/플래너 전략 비교
- 모델 비교: OpenAI/Gemini/Local Llama 등 모델별 성능·비용·안정성 비교
- 재현 실험: 논문 제공 스크립트로 합리/제한합리/민주적 시나리오 일괄 실행 및 추적

## 5. 기능 요구사항
- 시뮬레이션 엔진
  - 다중 에이전트(워커/플래너) 상호작용 루프, `--two-timescale`로 정책 업데이트 주기 제어
  - 메시지/행동/메트릭 기록과 상위 K 베스트 타임스텝 요약 제공
- 에이전트
  - `LLMAgent` 기반: IO/COT/SC 프롬프트 전략, 재시도·타임아웃, 히스토리 길이 관리
  - 워커/플래너 유형: `LLM`, `FIXED`, `US_FED`, `UNIFORM`
  - 페르소나: 인구통계 기반 샘플링 + LLM 생성, 현재 `bounded`/`democratic`에서 egotistical 지원
- 조세정책/브래킷
  - `bracket_setting`에 따른 브래킷 수와 기본 세율, 증감 폭(`delta`) 제한, 0~100% 범위 클리핑
- 모델 연동
  - OpenAI/OpenRouter/Gemini/vLLM/Ollama를 공통 인터페이스(`BaseLLMModel`)로 호출
  - 속성: `model_name`, `max_tokens`, `temperature`, `stop_tokens`
  - rate limit 백오프, JSON 응답 추출/검증
- 히스토리 관리
  - LLM 상호작용 히스토리: JSONL 저장/로드/리플레이(`enable_history_replay`)
  - 메시지 히스토리: per-agent JSON 저장, 재시작 시 자동 로드, 체크포인트(History replay step로 원하는 스텝)에서 재개
    예를 들어, 만약 History replay step이 5라면 worker 0는 worker_0_messages.json의 step 5에서의 User prompt와 System Prompt가 복구되어야 하고 time step은 History replay step인 5부터 시작되어야 함.
    다른 agent들도 각각의 json 파일에서 history replay step이 복원됨.
- 실험/예제
  - `examples/quick_start.py`, `examples/advanced_usage.py`로 기능/시나리오 검증
  - `experiments/run_experiments.py`로 사전 구성 실험 일괄 실행 및 W&B 연동 옵션
- UI
  - `streamlit_app.py`로 파라미터 구성, 실행, 로그 모니터링, 설정 저장/불러오기

## 6. 비기능 요구사항
- 성능/확장성: 1000+ 에이전트 시뮬레이션을 Batch/비동기 호출로 견딜 수 있도록 모델 호출 병렬화(서비스별 rate limit 고려)
- 안정성: LLM JSON 파싱 실패 시 재시도/백오프, 최대 재귀/리트라이 한계 설정
- 재현성: 시드·설정 파일·실험 스크립트 제공, 히스토리 리플레이로 동일 결과 재현
- 보안: API 키 환경변수, 로컬 모델 옵션 제공, 로그에 민감정보 미기록
- 비용: `gpt-4o-mini` 등 비용 효율 모델 기본값, 로컬(vLLM/Ollama) 옵션 안내

## 7. 구성/인터페이스(요약)
- CLI 주요 인자
  - 시나리오: `--scenario {rational,bounded,democratic,fixed}`
  - 에이전트: `--num-agents`, `--worker-type`, `--planner-type`, `--percent-ego/alt/adv`
  - LLM: `--llm`, `--service {vllm,ollama}`, `--port`, `--prompt-algo {io,cot,sc}`
  - 실행: `--max-timesteps`, `--two-timescale`, `--wandb`
  - 기록: `--history-jsonl-save/load`, `--history-jsonl-step`, `--history-save-interval`
- 파이썬 인터페이스
  - `BaseLLMModel.send_msg(system_prompt, user_prompt, temperature, json_format)`
  - `LLMAgent.act_llm(timestep, keys, parse_func)` 및 프롬프트 헬퍼(`prompt_io/cot/sc`)

## 8. 데이터 및 메트릭
- 입력: 인구통계 샘플, LLM 프롬프트, 조세 브래킷·유틸리티 파라미터
- 출력: 에이전트 행위, 세율 변화, 수익/후생·효용 메트릭, 상위 타임스텝 로그
- 추적: 실행 설정, 모델/서비스, 토큰/비용(확장), W&B 로그(옵션)

## 9. 성공 기준 (수용 기준)
- 기본 시나리오 3종이 20~200 스텝에서 안정적으로 실행되고 JSON 파싱 실패 시 자동 복구
- 최소 3개 모델(OpenAI, OpenRouter(Claude/LLama), Local vLLM/Ollama) 실동작
- Streamlit UI로 기본 파라미터 조정 및 실행/기록 확인 가능
- 히스토리 저장/로드/리플레이 통해 같은 설정으로 결과 재현

## 10. 리스크 및 대응
- LLM 불안정/Rate limit: 백오프·재시도, 로컬 모델 대체, OpenRouter 확장
- JSON 파싱 실패: CoT 사전응답 + 후속 JSON 강제, 재귀적 재시도
- 비용 급증: 소형 모델 기본값, 타임스텝/에이전트 상한, 로컬 실행 권장
- 재현성 저하: 히스토리 기반 리플레이, 설정/시드 고정, 예제·테스트 유지

## 11. 마일스톤
- M1: MVP 완성(시나리오/모델 연동, 기본 UI, 기록/리플레이) — v0.1
- M2: 프롬프트 전략 확장 및 성능 튜닝 — v0.2
- M3: 분산 실행·비용/메트릭 대시보드 — v0.3
- M4: 커스텀 유틸리티·정책학습 템플릿 — v0.4

## 12. 오픈 이슈(초안)
- `LLMAgent.parse_tax`의 `delta` 파라미터 명시적 정의와 유효성 검사 강화
- `BaseLLMModel.restore_history_to_step`의 `breakpoint()` 제거 및 테스트 추가
- 모델별 토큰/비용 로깅 표준화 및 Streamlit 노출
- 시나리오별 성능 벤치마크 스크립트와 리포트 자동 생성
