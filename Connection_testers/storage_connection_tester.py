import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

# Option 1: Using Account Key (what you're trying)
account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

# Make sure these are not None
if not account_name or not account_key:
    print("Missing credentials in .env file!")
    print(f"Account Name: {'Set' if account_name else 'Missing'}")
    print(f"Account Key: {'Set' if account_key else 'Missing'}")
else:
    # Create BlobServiceClient
    account_url = f"https://{account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(
        account_url=account_url, 
        credential=account_key
    )
    try:
        # Test connection
        containers = blob_service_client.list_containers()
        for container in containers:
            print(f"Container: {container['name']}")
            container_client = blob_service_client.get_container_client(container['name'])
            blobs = list(container_client.list_blobs())
            
            # Count files per directory
            dir_counts = {}
            for blob in blobs:
                parts = blob['name'].split('/')
                if len(parts) > 1:
                    dir_name = parts[0]
                    dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1
            
            # Print subdirectories and file counts
            if dir_counts:
                print(f"  Subdirectories:")
                for dir_name in sorted(dir_counts.keys()):
                    print(f"    {dir_name}: {dir_counts[dir_name]} files")
            else:
                print(f"  Total files: {len(blobs)}")
    except Exception as e:
        print(f"Error: {e}")