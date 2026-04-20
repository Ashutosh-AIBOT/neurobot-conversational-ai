
import os
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

load_dotenv()

def push():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN not found in .env or environment variables.")
        return

    api = HfApi()
    repo_id = "Ashutosh-AIBOT/NeuroBot-Intelligence" # You can change this
    
    print(f"🚀 Creating/Accessing Space: {repo_id}...")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="space", space_sdk="streamlit", exist_ok=True)
        
        print("📤 Uploading files...")
        upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            token=token,
            ignore_patterns=[".venv", "venv_uv", "__pycache__", "*.db*", "*.log", ".git"]
        )
        print(f"✅ Success! Your space is live at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to push: {e}")

if __name__ == "__main__":
    push()
