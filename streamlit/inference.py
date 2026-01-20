import gradio as gr
import onnxruntime as ort
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


model_path = 'data_models_Dist_Model_ver1_final_model.onnx'
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def predict(image):
    """이미지를 입력받아 전처리 후 ONNX 모델로 추론합니다."""
    transform = A.Compose([
        A.Resize(450, 650),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    img_tensor = transform(image=image)['image']
    input_tensor = img_tensor.unsqueeze(0).numpy() # (1, 3, 450, 650)

    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    logits = session.run(None, ort_inputs)[0]
    
    # Softmax 및 라벨 매핑
    exp_logits = np.exp(logits - np.max(logits))
    probs = (exp_logits / np.sum(exp_logits))[0]
    
    labels = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']
    return {labels[i]: float(probs[i]) for i in range(4)}

# Gradio 인터페이스
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=4),
    title="Plant Pathology 2020 API",
    description="ResNeSt101 ONNX Model Implementation"
)

if __name__ == "__main__":
    demo.launch()