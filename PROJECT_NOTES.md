# PROJECT_NOTES

이 문서는 `stevehan310.github.io` 저장소의 구조를 정리한 노트입니다.

## 사이트 개요

- **테마**: [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) (Jekyll)
- **제목**: "Steve's ML Notes"
- **저자**: Steve Han (Data Scientist, Saint Louis, MO)
- **URL**: https://stevehan310.github.io
- **언어**: 한국어 (`ko_KR`)
- **댓글**: Disqus (`stevemlblog`)
- **분석**: Google Analytics (google-gtag)
- **성격**: 통계·머신러닝 개념을 정리하는 개인 학습 블로그

## 디렉터리 구조

| 경로 | 설명 |
|---|---|
| `_posts/` | 실제 발행된 블로그 글 (Markdown). 2024-02 ~ 2026-08 작성 |
| `notebooks/` | 포스트에 대응하는 Jupyter 노트북. 아직 포스트로 변환되지 않은 것도 존재 |
| `_pages/` | 검색, 카테고리/태그 아카이브 등 정적 페이지 |
| `_data/navigation.yml` | 사이트 메뉴 구조 정의 |
| `_data/ui-text.yml` | 테마 UI 텍스트(로컬라이제이션) |
| `_includes/` | 테마 컴포넌트 조각 (헤더, 사이드바, 댓글, 검색 등) |
| `_layouts/`, `_sass/` | 테마 레이아웃 및 스타일 |
| `assets/` | 이미지, JS 등 정적 리소스. 포스트 이미지는 `assets/images/<주제명>/` 하위에 저장 |
| `docs/` | Minimal Mistakes 테마 자체의 데모 사이트 소스. `_config.yml`에서 Jekyll 빌드 시 제외됨 (테마 참고용, 실제 사이트 콘텐츠 아님) |
| `md/`, `_md/` | (임시) 포스트로 아직 정리되지 않은 마크다운 초안을 두는 스테이징 폴더. 발행 후 삭제되므로 저장소에 상시 존재하지 않음 — 있다면 미발행 초안이 남아있다는 뜻 |

## 현재 발행된 포스트 (`_posts/`)

- 2024-02-06 GroupBy
- 2024-03-07 Python Tips
- 2024-03-17 Log-Likelihood Estimation
- 2024-03-26 Generalized Linear Models
- 2024-04-06 Diff-in-Diff Testing
- 2024-06-11 Linear Regression Model in PyTorch and TensorFlow
- 2026-08-02 Basic Neural Network from Scratch: NumPy vs PyTorch
- 2026-08-10 Implementing Gradient Descent From Scratch: NumPy vs PyTorch
- 2026-08-11 Mastering the Adam Optimizer: From Concept to NumPy & PyTorch Implementation

## 포스트로 아직 안 나온 노트북 (`notebooks/`)

- NER with LSTM (2024-07-03)
- Bayesian Linear Regression (From Scratch / with PyMC) (2024-04-06)
- Maximum Likelihood Estimation (2024-03-17)
- Logistic Regression in TensorFlow and PyTorch (2024-06-18)
- RNN (2024-06-26)
- Maximum a Posteriori Estimation (진행중, `_in_progress` 파일명, 2024-04-06)

## 워크플로우

두 가지 경로가 함께 쓰이고 있음:

1. **노트북 → 포스트** (기존 방식): Jupyter 노트북(`notebooks/`)으로 학습 내용을 정리 → 동일 날짜/제목으로 Jekyll 포스트(`_posts/`)로 변환하여 게시. 예: 2026-08-02 Neural Network 포스트.
2. **마크다운 초안 → 포스트 → (선택적으로) 노트북** (최근 방식): `md/` 또는 `_md/`에 마크다운 초안을 먼저 작성 → 프론트매터(`layout`, `title`, `author`, `tags`, `categories`)를 추가해 `_posts/YYYY-MM-DD-Title_With_Underscores.md`로 옮기고 초안 폴더는 삭제 → 필요하면 이후에 `notebooks/`에 동일 stem의 노트북을 새로 만들어 실제 실행 결과(outputs)를 캡처해 추가. 예: 2026-08-10 Gradient Descent 포스트(노트북 없음), 2026-08-11 Adam Optimizer 포스트(포스트 발행 후 노트북 추가).

공통 컨벤션:
- 각 포스트는 `_config.yml`의 `defaults`에서 `layout: single`, `toc: true`, `comments: true`, `search: true` 등을 상속하므로, 포스트별 프론트매터는 `title`/`author`/`tags`/`categories`만 넣으면 됨.
- 2026-08-10 포스트부터 본문 최상단에 `title`과 중복되는 `# 제목` H1 헤딩을 넣지 않는 것으로 컨벤션 정리됨.

## 참고

- 테마 자체 데모 콘텐츠(`docs/`)는 실제 사이트 빌드에서 제외되므로 편집 시 혼동 주의.
- 사이트 업데이트 상세 이력은 `Change.md` 참고 (이 문서는 구조 요약, `Change.md`는 날짜별 변경 로그).
