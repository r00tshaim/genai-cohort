import uvicorn
import hmac
import hashlib
import json
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import os
import httpx

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# GitHub Webhook Secret (get from environment variable for security)
# IMPORTANT: Replace 'your_super_secret_string' with a strong, unique secret
# and store it in a .env file or your environment variables.
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "null")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "null")
if GITHUB_WEBHOOK_SECRET == "null" or GITHUB_WEBHOOK_SECRET == "null":
    print("ERROR: GITHUB_WEBHOOK_SECRET or GITHUB_TOKEN is missing please set it as an environment variable or in a .env file.")
    exit(0)


async def verify_signature(request: Request):
    """
    Verifies the GitHub webhook signature.
    """
    # Get the raw body bytes directly from the request
    # FastAPI handles parsing the JSON payload separately
    body = await request.body()
    signature_header = request.headers.get('x-hub-signature-256')

    if not signature_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No signature header found")

    try:
        # Expected format: "sha256=<hex_digest>"
        sha_name, signature = signature_header.split('=')
        if sha_name != 'sha256':
            raise ValueError("Signature header is not sha256")

        mac = hmac.new(GITHUB_WEBHOOK_SECRET.encode('utf-8'), msg=body, digestmod=hashlib.sha256)
        # Compare the computed digest with the received signature
        if not hmac.compare_digest(mac.hexdigest(), signature):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signature mismatch")
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid signature format or secret: {e}")
    


async def get_commit_diffs(owner: str, repo: str, base_sha: str, head_sha: str) -> str:
    """
    Fetches the diff between two commits from GitHub API.
    """
    if not GITHUB_TOKEN:
        raise ValueError("GitHub Token is not configured.")

    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"
    headers = {
        "Accept": "application/vnd.github.v3.diff", # Request the diff format
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    print("Shaim url:",url)

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.text # Returns the raw diff string

def parse_git_diff(diff_text: str) -> dict[str, str]:
    """
    Parses a unified git diff string into a dictionary of file paths and their diffs.
    This is a simplified parser and might need adjustments for complex diffs.
    """
    files = {}
    current_file = None
    current_diff_lines = []

    lines = diff_text.splitlines()
    for line in lines:
        if line.startswith("diff --git"):
            if current_file and current_diff_lines:
                files[current_file] = "\n".join(current_diff_lines).strip()
            
            # Extract file path: diff --git a/old_path b/new_path
            # We usually care about the 'new_path'
            parts = line.split(" ")
            if len(parts) >= 4:
                # Remove "b/" prefix if present
                current_file = parts[3].lstrip('b/')
            else:
                current_file = "UNKNOWN_FILE_PATH" # Fallback
            current_diff_lines = [line] # Start with the 'diff --git' line itself
        else:
            current_diff_lines.append(line)

    # Add the last file
    if current_file and current_diff_lines:
        files[current_file] = "\n".join(current_diff_lines).strip()

    return files



async def get_pull_request_diff(diff_url: str) -> str:
    """
    Fetches the full diff for a Pull Request directly using its diff_url.
    """
    if not GITHUB_TOKEN:
        raise ValueError("GitHub Token is not configured.")

    print(f"Attempting to fetch PR diff from URL: {diff_url}")
    headers = {
        "Accept": "application/vnd.github.v3.diff", # Request raw diff format
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(diff_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch PR diff from {diff_url}. Status: {response.status_code}, Response: {response.text}")
        response.raise_for_status()
        return response.text # This will be the raw diff string


@app.post("/webhook", dependencies=[Depends(verify_signature)])
async def github_webhook(request: Request):
    """
    Handles incoming GitHub webhook events.
    """
    event = request.headers.get('x-github-event')
    if not event:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No 'X-GitHub-Event' header found")

    payload = await request.json()

    print(f"Received GitHub event: {event}")
    print('Payload:', json.dumps(payload, indent=2))

    # Handle different event types
    if event == 'push':
        repo_full_name = payload.get('repository', {}).get('full_name', 'N/A')
        pusher_name = payload.get('pusher', {}).get('name', 'N/A')
        print(f"Push event to {repo_full_name} by {pusher_name}")

        before_sha = payload.get('before')
        after_sha = payload.get('after')

        if not all([repo_full_name, before_sha, after_sha]):
            print("Missing essential data in push event payload.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incomplete push event payload.")
        
        print(f"Push event to {repo_full_name} by {pusher_name}")
        print(f"Comparing {before_sha} (before) to {after_sha} (after)")

        owner, repo_name = repo_full_name.split('/')

        if GITHUB_TOKEN:
            try:
                print("Shaim 0")

                diffs_data = await get_commit_diffs(owner, repo_name, before_sha, after_sha)

                print("Shaim 1")
                
                # Process the diffs_data
                parsed_diffs = parse_git_diff(diffs_data)

                print("Shaim 2")
                
                for file_path, file_diff in parsed_diffs.items():
                    print("Shaim 3")
                    print(f"File: {file_path}")
                    # logger.info(f"Diff:\n{file_diff}") # Uncomment to log full diff for each file
                    # You can now process each file_path and its file_diff here
                    # For example, send it to an AI model for review
                    print(f"\n--- File: {file_path} ---")
                    print(file_diff)

            except Exception as e:
                print(f"Failed to get commit diffs: {e}")
                # You might still return 200 OK to GitHub, but log the error
        else:
            print("Cannot retrieve diffs: GITHUB_TOKEN is not set.")

        # Example: trigger a local build script
        # subprocess.run(["./your_build_script.sh"])
    elif event == 'pull_request':
        print("Received a pull_request event.")
        pull_request_data = payload.get("pull_request")
        if pull_request_data:
            diff_url = pull_request_data.get("diff_url")
            owner = pull_request_data["base"]["repo"]["owner"]["login"]
            repo_name = pull_request_data["base"]["repo"]["name"]
            pr_number = pull_request_data["number"]

            if diff_url and GITHUB_TOKEN:
                try:
                    full_pr_diff = await get_pull_request_diff(diff_url)
                    print(f"Successfully fetched diff for PR #{pr_number} in {owner}/{repo_name}.")
                    print(f"\n--- Full Diff for PR #{pr_number} ({owner}/{repo_name}) ---")
                    print(full_pr_diff)
                    # Here you would typically process this full_pr_diff
                    # e.g., pass it to your LLM or analyze changes.

                except Exception as e:
                    print(f"Error fetching/processing PR diff for PR #{pr_number}: {e}")
            else:
                print("PR diff_url not found or GITHUB_TOKEN not set for pull_request event.")
        else:
            print("Pull request data missing from payload for pull_request event.")
        # Example: run tests
        # subprocess.run(["pytest", "your_tests/"])
    elif event == 'ping':
        print("Received ping event from GitHub.")
    else:
        print(f"Unhandled event type: {event}")

    return PlainTextResponse("Webhook received successfully!", status_code=status.HTTP_200_OK)

# To run the application locally:
# uvicorn main:app --reload --port 8000
# Then use ngrok to expose it: ngrok http 8000