---
title: Plant Pathology Diagnosis Demo
emoji: 🌿
colorFrom: green
colorTo: indigo
sdk: static
pinned: false
short_description: 사과나무 잎 병해 4-클래스 진단 인터랙티브 데모
---

식물 잎 병해 진단(Plant Pathology 2020) 프로젝트의 인터랙티브 데모입니다.

- 샘플 5장은 사전 계산된 4-클래스 확률을 즉시 표시
- 업로드 이미지는 `ingyoun/plant-pathology-api`(ResNeSt101e ONNX)를 실시간 호출
- 포트폴리오 상세 페이지에 iframe으로 임베드되며 `postMessage`로 높이를 동기화

## 배포

이 폴더를 HF static space `ingyoun/plant-pathology-demo`에 업로드하면 갱신됩니다.

```python
from huggingface_hub import upload_folder
upload_folder(folder_path=".", repo_id="ingyoun/plant-pathology-demo", repo_type="space")
```

서빙 URL: https://ingyoun-plant-pathology-demo.static.hf.space
