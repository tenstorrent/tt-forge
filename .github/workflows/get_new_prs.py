import subprocess
import json
import os
import urllib.request
import sys

def checkTT(author_login):
    print(f"DEBUG: Checking if {author_login} is a member of the organization.")
    # Define the API endpoint and headers
    token = os.getenv("ORG_READ_GITHUB_TOKEN")
    url = f"https://api.github.com/orgs/tenstorrent/members/{author_login}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Python-urllib"
    }
    
    # Make the API request
    request = urllib.request.Request(url, headers=headers)

    # try: Disabled, let it fail if http error happens
    with urllib.request.urlopen(request) as response:
        print(f"DEBUG: Received response status {response.status} for {author_login}.")
        # Check the HTTP status code
        return (response.status == 204)
    # except urllib.error.HTTPError as e:
    #     print(f"DEBUG: HTTPError while checking {author_login}: {e}")
    #     return False

def fetch_and_filter_prs(last, repo):
    print(f"DEBUG: Fetching PRs for repo {repo} with last cursor: {last}")
    query = """
    query($last:String,$repo:String!) {
        repo: repository(owner: "tenstorrent", name: $repo) {
            pullRequests(states: OPEN, first: 100,  after: $last) {
                nodes {
                    title
                    url
                    createdAt
                    author {
                        login
                    }
                }
                pageInfo {
                    endCursor
                    hasNextPage
                    hasPreviousPage
                }
            }
        }
    }
    """

    # Execute the `gh api graphql` command
    command = [
        "gh", "api", "graphql", 
        "-F", f"last={last}", 
        "-F", f"repo={repo}", 
        "-f", f"query={query}"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to execute GraphQL query. stderr: {result.stderr}")
        sys.exit(result.returncode)

    # Parse the JSON response
    response = json.loads(result.stdout)
    prs = response.get("data", {}).get("repo", {}).get("pullRequests", {})
    print(f"DEBUG: Retrieved {len(prs.get('nodes', []))} PRs for repo {repo}.")
    
    # Filter PRs and build the result
    result = { prs: [] }
    for pr in prs.get("nodes", []):
        author_login = pr.get("author", {}).get("login")
        print(f"DEBUG: Processing PR by {author_login}.")
        if not checkTT(author_login):
            print(f"DEBUG: {author_login} is not a member. Adding PR to results.")
            result.prs.append({
                "url": pr["url"],
                "title": pr["title"],
                "createdAt": pr["createdAt"],
                "author": author_login
            })

    # Add the endCursor value
    print(f"DEBUG: Updated last cursor for {repo}: {prs.get('pageInfo', {}).get('endCursor')}")
    result["last"] = prs.get("pageInfo", {}).get("endCursor")
    
    return result

def send_message(message):
    print(f"DEBUG: Sending message to Slack: {message}")
    # Define the Slack webhook URL
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("ERROR: SLACK_WEBHOOK_URL is not set.")
        return

    # Prepare the payload
    payload = json.dumps({"text": message}).encode("utf-8")

    # Send the message to the Slack webhook
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            print(f"DEBUG: Slack message sent with response status {response.status}.")
            if response.status != 200:
                print(f"Failed to send message: {response.status}")
    except urllib.error.HTTPError as e:
        print(f"DEBUG: HTTPError while sending Slack message: {e}")

def load_and_run(json_file):
    print(f"DEBUG: Loading and running with JSON file: {json_file}")
    # Check if the JSON file exists, if not create it with default values
    if not os.path.exists(json_file):
        print(f"DEBUG: JSON file {json_file} does not exist. Creating default data.")
        data = {
            "tt-mlir": {"last": ""},
            "tt-xla": {"last": ""},
            "tt-forge-fe": {"last": ""},
            "tt-torch": {"last": ""}
        }
    else:
        print(f"DEBUG: JSON file {json_file} found. Loading data.")
        with open(json_file, "r") as file:
            data = json.load(file)

    # Process each repository
    for repo, info in data.items():
        print(f"DEBUG: Processing repository {repo}.")
        last = info.get("last", "")
        result = fetch_and_filter_prs(last, repo)
        data[repo]["last"] = result.get("last", "")

        # Send messages for each PR
        for pr in result.get("prs", []):
            print(f"DEBUG: Sending Slack message for PR: {pr['url']}")
            message = f"New PR in {repo}:\nTitle: {pr['title']}\nURL: {pr['url']}\nAuthor: {pr['author']}\nCreated At: {pr['createdAt']}"
            send_message(message)

    # Save the updated JSON back to the file
    print(f"DEBUG: Saving updated data to {json_file}.")
    with open(json_file, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    print("DEBUG: Script started.")
    if len(sys.argv) != 2:
        print("DEBUG: Invalid arguments. Exiting.")
        print("Usage: python get_new_prs.py <json_file>")
    else:
        print(f"DEBUG: Running with argument: {sys.argv[1]}")
        load_and_run(sys.argv[1])
