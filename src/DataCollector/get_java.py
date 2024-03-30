import requests
import re
import xml.etree.ElementTree as ET
import os
import base64
import shutil
from tqdm import tqdm
from github import Github
from github import GithubException
import subprocess
import time

pattern = r"(?im)log.*\.(log|error|info|warn|fatal|debug|trace|off|all)\(.*\)"
regex = re.compile(pattern, re.DOTALL)


def git_clone(owner, repo):
    max_attempts = 5
    retry_wait_time = 5  # in seconds

    git_url = f"https://github.com/{owner}/{repo}.git"
    local_path = f"./temp/{repo}"
    cmd = ["git", "clone", git_url, local_path]

    for i in range(max_attempts):
        try:
            subprocess.check_call(cmd)
            print("Git clone successful!")
            break
        except subprocess.CalledProcessError as e:
            print(f"Git clone attempt {i + 1} failed with error code {e.returncode}. Retrying in {retry_wait_time} seconds...")
            time.sleep(retry_wait_time)
    else:
        print(f"Git clone failed after {max_attempts} attempts.")


def get_sha_for_tag(repository, tag):
    """
    Returns a commit PyGithub object for the specified repository and tag.
    """
    branches = repository.get_branches()
    matched_branches = [match for match in branches if match.name == tag]
    if matched_branches:
        return matched_branches[0].commit.sha

    tags = repository.get_tags()
    matched_tags = [match for match in tags if match.name == tag]
    if not matched_tags:
        print("No Tag or Branch exists with that name")
        return None
    return matched_tags[0].commit.sha


def check_java(path):
    try:
        with open(path, 'r') as file:
            content = file.read()
            words = content.split()
            if len(words) > 300:
                return False
            lines = content.split('\n')
            if len(lines) > 300:
                return False
            match = regex.search(content)
            if match:
                return True
    except UnicodeDecodeError as e:
        print(f"Error: {e} and Path: {path}")
        return False
    return False


def download_java_file(git, sha, repo, path):
    try:
        file_content = git.get_contents(path, ref=sha)
        _, file_name = os.path.split(path)
        file_data = base64.b64decode(file_content.content)
        file_out = open(f"repos/{repo}/{file_name}", "wb")
        file_out.write(file_data)
        file_out.close()
        if check_java(f"repos/{repo}/{file_name}") == False:
            os.remove(f"repos/{repo}/{file_name}")
            return 0
        return 1
    except (GithubException, IOError) as exc:
        print('Error processing %s: %s', path, exc)
        return 0


def download_java(owner, repo, access_token, branch="master"):
    if not os.path.exists(f"repos/{repo}/"):
        os.makedirs(f"repos/{repo}/")

    git = Github(access_token)
    try:
        git_repo = git.get_repo(f"{owner}/{repo}")
    except GithubException as e:
        if e.status == 404:
            print("Non")
        else:
            print("Error")
        shutil.rmtree(f"repos/{repo}/")
        return False
    sha = get_sha_for_tag(git_repo, branch)

    # Define the Github Tree API endpoint and repository details
    api_url = 'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'
    # Make an HTTP GET request to the Github Tree API endpoint
    headers = {'Authorization': f'token {access_token}'}
    response = requests.get(api_url.format(owner=owner, repo=repo, branch=branch), headers=headers)
    data = response.json()
    cnt1, cnt2 = 0, 0
    print(git_repo.size)
    if git_repo.size < 500000000:
        git_clone(owner, repo)
        for subdir, dirs, files in os.walk(f"./temp/{repo}"):
            for file in tqdm(files):
                if not file.endswith(".java"):
                    continue
                file_path = os.path.join(subdir, file)
                if os.path.getsize(file_path) < 15 * 1024:
                    cnt2 += 1
                    if check_java(file_path):
                        cnt1 += 1
                        shutil.copy2(file_path, f"repos/{repo}/{file}")
        shutil.rmtree(f"./temp/{repo}")
    else:
        print("File is too large!")
        if sha is not None:
            tree = data['tree']
            leng = len(tree)
            for file in tqdm(tree):
            #for item in tqdm(tree):
                if file['type'] != "tree" and file['size'] < 15 * 1024 and file['path'].endswith(".java"):
                    cnt1 += 1
                    cnt1 += download_java_file(git_repo, sha, repo, file['path'])
                    cnt2 += 1
    if cnt1 == 0:
        shutil.rmtree(f"repos/{repo}/")
    print(f"{owner}/{repo} downloaded: {cnt1}/{cnt1+cnt2} files")
    return cnt1, cnt2
