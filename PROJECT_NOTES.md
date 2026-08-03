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
| `_posts/` | 실제 발행된 블로그 글 (Markdown). 2024-02 ~ 2024-06 작성 |
| `notebooks/` | 포스트에 대응하는 Jupyter 노트북 원본. 아직 포스트로 변환되지 않은 것도 존재 |
| `_pages/` | 검색, 카테고리/태그 아카이브 등 정적 페이지 |
| `_data/navigation.yml` | 사이트 메뉴 구조 정의 |
| `_data/ui-text.yml` | 테마 UI 텍스트(로컬라이제이션) |
| `_includes/` | 테마 컴포넌트 조각 (헤더, 사이드바, 댓글, 검색 등) |
| `_layouts/`, `_sass/` | 테마 레이아웃 및 스타일 |
| `assets/` | 이미지, JS 등 정적 리소스 |
| `docs/` | Minimal Mistakes 테마 자체의 데모 사이트 소스. `_config.yml`에서 Jekyll 빌드 시 제외됨 (테마 참고용, 실제 사이트 콘텐츠 아님) |

## 현재 발행된 포스트 (`_posts/`)

- 2024-02-06 GroupBy
- 2024-03-07 Python Tips
- 2024-03-17 Log-Likelihood Estimation
- 2024-03-26 Generalized Linear Models
- 2024-04-06 Diff-in-Diff Testing
- 2024-06-11 Linear Regression Model in PyTorch and TensorFlow

## 포스트로 아직 안 나온 노트북 (`notebooks/`)

- NER with LSTM
- Bayesian Linear Regression (From Scratch / with PyMC)
- Maximum Likelihood Estimation
- Logistic Regression in TensorFlow and PyTorch
- RNN
- Maximum a Posteriori Estimation (진행중, `_in_progress` 파일명)

## 워크플로우 추정

Jupyter 노트북(`notebooks/`)으로 학습 내용을 정리 → 동일 날짜/제목으로 Jekyll 포스트(`_posts/`)로 변환하여 블로그에 게시하는 방식으로 운영.

## 참고

- `_config.yml`의 `defaults` 설정에 따라 모든 포스트는 `layout: single`, `toc: true`(목차 사이드바 고정), `comments: true`, `search: true` 등이 기본 적용됨.
- 테마 자체 데모 콘텐츠(`docs/`)는 실제 사이트 빌드에서 제외되므로 편집 시 혼동 주의.
