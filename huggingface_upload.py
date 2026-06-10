from huggingface_hub import HfApi
import os

api = HfApi()

# 1. 레포 생성 (한 번만 실행)
api.create_repo(
    repo_id="franktome/dreamerv3-custom-envs",  # 본인 HF 아이디로 변경
    repo_type="model",
    private=False,
    exist_ok=True
)
print("레포 생성 완료!")

# 2. 각 환경 체크포인트 업로드
BASE = "/mnt/hdd/hyeonseo/workspace/dreamerv3/logdir"

envs = [
    "highway_highway",
    "highway_intersection",
    "highway_merge",
    "highway_roundabout",
    "highway_train_v1",
]

for env in envs:
    ckpt_dir = f"{BASE}/{env}/ckpt"
    print(f"\n[{env}] 업로드 중...")
    
    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id="franktome/dreamerv3-custom-envs",  # 본인 HF 아이디로 변경
        path_in_repo=f"checkpoints/{env}",
        repo_type="model"
    )
    print(f"[{env}] 업로드 완료!")

print("\n모든 체크포인트 업로드 완료!")
print("확인: https://huggingface.co/hyeonseo/dreamerv3-custom-envs")