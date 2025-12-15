import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient

load_dotenv()

def check_openai():
    try:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        print("[PASS] Azure OpenAI\n")
        return True
    except Exception as e:
        print(f"[FAIL] Azure OpenAI: {e}\n")
        return False

def check_storage():
    try:
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
        
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        containers = list(blob_service.list_containers())
        print(f"[PASS] Azure Storage: {len(containers)} containers\n")
        return True
    except Exception as e:
        print(f"[FAIL] Azure Storage: {e}\n")
        return False

def check_env_vars():
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "DEPLOYMENT_NAME",
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_ACCOUNT_KEY"
    ]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print(f"[FAIL] Missing: {missing}\n")
        return False
    print("[PASS] Environment variables\n")
    return True

def run_health_check():
    print("CropWatch-AI Health Check\n")
    results = [check_env_vars(), check_openai(), check_storage()]
    passed = sum(results)
    print(f"Result: {passed}/{len(results)} passed\n")
    return passed == len(results)

if __name__ == "__main__":
    run_health_check()