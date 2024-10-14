import torch
from pathlib import Path

def load_model_and_print_step(run_id: str, models_dir: Path):
    """
    모델 경로를 제공받아 스텝 수를 출력하는 함수.
    
    Args:
        run_id: 모델의 식별자 (모델 파일명에 사용됨)
        models_dir: 모델이 저장된 디렉토리 경로
        
    Returns:
        없음
    """
    # 모델 파일 경로 설정
    state_fpath = models_dir.joinpath(run_id + ".pt")

    # 모델 경로가 유효한지 확인
    if not state_fpath.exists():
        print(f"No model found at the specified path: {state_fpath}")
        return

    # 기존 모델 체크포인트 로드
    print(f"Found existing model at \"{state_fpath}\", loading it to check the training step.")
    checkpoint = torch.load(state_fpath)

    # 스텝 수 출력
    init_step = checkpoint.get("step", None)
    if init_step is not None:
        print(f"Model was last trained until step: {init_step}")
    else:
        print("Step information not found in the checkpoint.")

# Usage example
run_id = "my_second_model"
models_dir = Path(r"C:\Users\admin\Desktop\sv2tts_korean\encoder\saved_models")
load_model_and_print_step(run_id, models_dir)
