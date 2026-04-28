import requests
import base64
import os
import time
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(dotenv_path="/Users/abdullahnaeem/Projects/distributed-urdu-ocr/deployment/hadoop/.env")

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_URL = "https://api.runpod.ai/v2/z3zabzqi52jyoh/run"

def run_inference(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return

    # 1. Encode image to Base64
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "image": base64_string
        }
    }

    # 2. Start the job
    print(f"Sending request for {image_path}...")
    response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"Error starting job: {response.text}")
        return

    job_data = response.json()
    job_id = job_data.get("id")
    print(f"Job started. ID: {job_id}")

    # 3. Poll for result
    status_url = f"https://api.runpod.ai/v2/z3zabzqi52jyoh/status/{job_id}"
    
    print("Waiting for result (this may take 1-2 mins if worker is cold)...")
    while True:
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()
        status = status_data.get("status")

        if status == "COMPLETED":
            result = status_data.get("output")
            print("\n--- OCR OUTPUT ---")
            print(result)
            print("------------------\n")
            break
        elif status == "FAILED":
            print(f"Job failed: {status_data.get('error')}")
            break
        
        time.sleep(2)  # Wait 2 seconds before polling again

if __name__ == "__main__":
    test_image = "line1-urdu.png"
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    run_inference(test_image)
