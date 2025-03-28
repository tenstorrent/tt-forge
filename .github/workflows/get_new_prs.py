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

def fetch_and_filter_prs(repo, key, last):
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
    for pr in prs.get("nodes", []):
        author_login = pr.get("author", {}).get("login")
        print(f"DEBUG: Processing PR by {author_login}.")
        if not checkTT(author_login):
            print(f"DEBUG: {author_login} is not a member. Sending message.")
            message = f"New PR in {repo}:\nTitle: {pr['title']}\nURL: {pr['url']}\nAuthor: {pr['author']}\nCreated At: {pr['createdAt']}"
            send_message(message)
        else:
            print(f"DEBUG: {author_login} is a member.\nPR in {repo}:\nTitle: {pr['title']}\nURL: {pr['url']}\nAuthor: {pr['author']}\nCreated At: {pr['createdAt']}")

    # Add the endCursor value
    last = prs.get("pageInfo", {}).get("endCursor")
    
    # Update the GitHub variable with the new cursor value
    command = [
        "gh", "variable", "set", key, "--body", f"{last}"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to update GitHub variable. stderr: {result.stderr}")

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


if __name__ == "__main__":
    print("DEBUG: Script started.")
    if len(sys.argv) != 3:
        print("DEBUG: Invalid arguments. Exiting.")
        print("Usage: python get_new_prs.py <repo> <key> <last>")
    else:
        print(f"DEBUG: Running with arguments: {sys.argv[1]}, {sys.argv[2]} {sys.argv[3]}")
        fetch_and_filter_prs(sys.argv[1], sys.argv[2], sys.argv[3])
