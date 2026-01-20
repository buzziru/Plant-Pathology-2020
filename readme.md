# Plant Pathology 2020 AI Diagnosis

> **Deployment URL**: [https://kaggle-plant.streamlit.app/](https://kaggle-plant.streamlit.app/)

## 1. 프로젝트 개요 (Project Overview)

본 프로젝트는 **Kaggle Plant Pathology 2020** 경진대회의 데이터셋을 활용하여 사과나무 잎의 건강 상태를 진단하는 AI 서비스를 구축하는 것을 목표로 합니다. 컴퓨터 비전 기술을 통해 잎의 이미지를 분석하고, 다음 4가지 상태 중 하나로 정밀하게 분류합니다.

*   **Healthy**: 건강한 상태
*   **Multiple Diseases**: 복합 질병 상태
*   **Rust**: 녹병 (Rust)
*   **Scab**: 붉은곰팡이병 (Scab)

단순한 모델 학습을 넘어, **ResNeSt** 아키텍처 도입과 **지식 증류(Knowledge Distillation)** 기법을 통해 성능을 극대화하였으며, **ONNX Runtime**으로 추론 속도를 최적화하여 실제 웹 서비스로 배포하는 Full-Stack AI 파이프라인을 구현하였습니다.

---

## 2. 기술적 핵심 (Technical Core)

### 2.1 Model Architecture: ResNeSt (Split-Attention Networks)

본 프로젝트에서는 기본 ResNet 대신 **ResNeSt**를 백본으로 채택하였습니다. ResNeSt는 **Split-Attention Block**이라는 새로운 모듈을 도입하여 이미지의 채널 간 상호의존성을 효과적으로 모델링하는 아키텍처입니다.

*   **Split-Attention**: 입력 피처맵을 여러 그룹(Cardinality)과 서브 그룹(Radix)으로 나누어 처리한 후, 어텐션 메커니즘을 통해 다시 통합합니다.
*   **Feature-map Attention**: 채널 별 중요도를 동적으로 조정하여, 잎의 병변과 같이 미세하고 복잡한 패턴을 포착하는 데 탁월한 성능을 보입니다.

이러한 구조적 특징 덕분에 ResNeSt는 동일한 파라미터 수 대비 더 높은 정확도와 강건성(Robustness)을 제공하므로, 미세한 병변 구분이 필요한 식물 병리학 진단에 최적이라 판단하였습니다.

### 2.2 Training Strategy: Knowledge Distillation

모델의 일반화 성능을 높이고 과적합을 방지하기 위해 **지식 증류(Knowledge Distillation)** 전략을 적용하였습니다. 이는 크고 강력한 Teacher 모델의 지식(Dark Knowledge)을 경량화된 Student 모델(ResNeSt)에 전이하는 과정입니다.

Student 모델은 실제 정답 라벨(Hard Label)뿐만 아니라, Teacher 모델이 예측한 확률 분포(Soft Label)를 동시에 학습합니다. 이 과정에서 사용된 **Loss Function**은 다음과 같이 정의됩니다.

$$
L = \alpha L_{soft} + (1-\alpha) L_{hard}
$$

여기서 각 항의 의미는 다음과 같습니다.

*   $L_{soft}$: Teacher 모델과 Student 모델의 출력(Logits) 간의 불일치를 측정하는 손실 함수(KL Divergence 사용). Teacher가 바라보는 클래스 간의 관계를 학습합니다.
*   $L_{hard}$: Student 모델의 예측값과 실제 정답(Ground Truth) 간의 Cross-Entropy Loss.
*   $\alpha$: 두 손실 함수 간의 가중치를 조절하는 하이퍼파라미터(GT의 비율)

이 수식을 통해 Student 모델은 정답을 맞추는 것뿐만 아니라, Teacher 모델이 갖고 있는 '오답에 대한 정보'까지 함께 학습하여 더 강건한 결정 경계를 형성하게 됩니다.

> **Performance Boost**: 특히 본 프로젝트에서는 **ConvNeXt**를 Teacher 모델로, **ResNeSt**를 Student 모델로 설정하여 실험한 결과, 단일 모델 학습 대비 폭발적인 성능 향상을 확인하였습니다.

### 2.3 Optimization: ONNX Runtime

학습된 PyTorch 모델을 배포 단계에서 그대로 사용하지 않고, **ONNX** 포맷으로 변환하여 최적화를 수행하였습니다.

---

## 3. 개발 워크플로우 및 주요 파일 (Development Workflow)

전체 개발 과정은 실험, 모델 구성, 최적화, 그리고 서비스 배포의 단계로 구성되어 있습니다.

*   **실험 및 학습 (Training)**:
    *   `notebooks/26_00-plant-student14-resnest.ipynb`: 실험적 노트북 코드입니다.
    *   **`src/` 모듈 (Refactoring)**: `resnest101e` 기반의 핵심 학습 로직을 **PyTorch Lightning** 프레임워크를 활용하여 클래스 단위로 모듈화하였습니다. 
        *   `model.py`: `LightningModule`을 상속받아 모델 구조, 손실 함수, 최적화 로직을 캡슐화.
        *   `dataset.py`: `LightningDataModule`을 상속받아 데이터 전처리 및 로딩 파이프라인 관리.
        *   `runner.py`: 실험 전체를 관장하는 실행기.
        이를 통해 코드의 재사용성과 유지보수성을 대폭 향상시켰습니다. 
*   **배포용 모델 구성 (Model Setup)**:
    *   `notebooks/00_00-plant-Dist_Model.ipynb`: 지식 증류를 위한 Teacher-Student 구조 설정 및 사전 학습된 가중치 로드 등을 처리합니다.
*   **추론 최적화 (Optimization)**:
    *   `notebooks/transform_onnx.ipynb`: 학습된 PyTorch 모델(.pth)을 ONNX 포맷(.onnx)으로 변환하고, 추론 결과를 검증하는 로직이 구현되어 있습니다.
*   **백엔드 (API Serving)**:
    *   `inference.py`: Hugging Face Spaces 환경에서 Gradio를 기반으로 구동되며, ONNX 모델을 로드하여 실제 추론을 수행하는 API 서버 역할을 합니다.
*   **프론트엔드 (Dashboard)**:
    *   `app.py`: Streamlit을 활용한 웹 인터페이스입니다. 사용자가 이미지를 업로드하면 백엔드 API와 통신하여 결과를 시각화합니다.

---

## 4. 프로젝트 구조 (Project Structure)

```
my-project/
├── data/             # 데이터셋 (Git 제외됨)
├── notebooks/        # EDA 및 모델 학습 과정 (.ipynb)
├── src/              # 핵심 로직 (Python Modules)
│   ├── __init__.py
│   ├── config.py     # 파라미터 및 경로 설정 (CFG)
│   ├── dataset.py    # 데이터 로더 및 전처리 (DataModule)
│   ├── model.py      # ResNeSt 모델 및 손실 함수
│   ├── runner.py     # 실험 실행 및 학습 루프 (ExperimentRunner)
│   └── utils.py      # 유틸리티 함수 (Seed, Metrics)
├── streamlit/        # app.py, inference.py 및 샘플 이미지
├── requirements.txt  # 의존성 목록
└── README.md         # 프로젝트 문서
```

---

## 5. 서비스 작동 흐름 (Service Flow)

사용자가 웹 대시보드에 접속하여 진단을 수행하는 과정은 다음과 같습니다.

1.  **이미지 업로드 (User Action)**: Streamlit 웹 인터페이스(`app.py`)를 통해 사용자가 식물 잎 이미지를 업로드합니다.
2.  **전처리 및 요청 (Preprocessing)**: 이미지를 모델 입력 크기에 맞게 리사이징 및 정규화(Normalization)한 후, 백엔드로 전송합니다.
3.  **AI 추론 (Inference)**: `inference.py`에서 ONNX Runtime 세션을 통해 고속 추론을 수행합니다. 4가지 클래스에 대한 확률값을 계산합니다.
4.  **결과 반환 및 시각화 (Visualization)**: 가장 높은 확률을 가진 상태를 진단 결과로 사용자 화면에 출력하며, 각 상태별 확신도(Confidence)를 그래프로 보여줍니다.

---

## 참고 자료 (References)

*   **Technical Blog**: [캐글 Plant Pathology 2020를 정리하며: 지식 증류 전략](https://velog.io/@masew8/캐글-Plant-Pathology-2020를-정리하며-지식-증류-전략)
*   **Dataset**: Kaggle Plant Pathology 2020 - FGVC7
