import requests
import re
import xml.etree.ElementTree as ET
import os
import base64
import shutil
from github import Github
from github import GithubException

def check_string_in_file(file_path, search_str="log4j"):
    with open(file_path, 'r') as file:
        content = file.read()

    if "log4j" in content.lower() or "slf4j" in content.lower():
        return True
    else:
        return False

def check_log4j(pom_file_path):
    # Parse the POM file as XML
    try:
        # Parse XML file
        tree = ET.parse(pom_file_path)
        root = tree.getroot()

        # Define the Log4j dependency artifact details
        group_id = 'org.apache.logging.log4j'
        artifact_id = 'log4j-core'

        # Iterate over the dependency elements in the POM file and check for the Log4j dependency
        for dependency in root.findall('.//{http://maven.apache.org/POM/4.0.0}dependency'):
            # Retrieve the group ID and artifact ID of the dependency
            dep_group_id = dependency.find('.//{http://maven.apache.org/POM/4.0.0}groupId')
            dep_artifact_id = dependency.find('.//{http://maven.apache.org/POM/4.0.0}artifactId')
            if dep_group_id is not None and dep_artifact_id is not None:
                dep_group_id, dep_artifact_id = dep_group_id.text, dep_artifact_id.text
                # Check if the dependency is the Log4j dependency
                if dep_group_id == group_id and dep_artifact_id == artifact_id:
                    print(f'The POM file {pom_file_path} features the Log4j dependency')
                    return True

    except ET.ParseError as e:
        # Handle XML parsing exception
        print('Error parsing XML file:', e)
    
    print(f'The POM file {pom_file_path} does not feature the Log4j dependency')
    return False


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


def download_file(git, sha, repo, path):
    try:
        file_content = git.get_contents(path, ref=sha)
        file_data = base64.b64decode(file_content.content)
        directory_path, _ = os.path.split(path)
        if not os.path.exists(f"repos/{repo}/{directory_path}"):
            os.makedirs(f"repos/{repo}/{directory_path}", exist_ok=True)
        file_out = open(f"repos/{repo}/{path}", "wb")
        file_out.write(file_data)
        file_out.close()
    except (GithubException, IOError) as exc:
        print('Error processing %s: %s', path, exc)

def check_repo(owner, repo, branch="master"):
    # Define the Github Tree API endpoint and repository details
    api_url = 'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'
    # Make an HTTP GET request to the Github Tree API endpoint
    access_token = "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"
    headers = {'Authorization': f'token {access_token}'}
    if not os.path.exists(f"repos/{repo}/"):
        os.makedirs(f"repos/{repo}/")
        #print(f"./{repo}/ created")

    git = Github("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")
    git_repo = git.get_repo(f"{owner}/{repo}")
    sha = get_sha_for_tag(git_repo, branch)
    # Parse the response data as JSON
    response = requests.get(api_url.format(owner=owner, repo=repo, branch=branch), headers=headers)
    data = response.json()
    contain_pom = False
    if sha is not None:
        for item in data['tree']:
            if re.search("pom.xml", item['path'], re.IGNORECASE):
                download_file(git_repo, sha, repo, item['path'])
                if check_log4j(f"repos/{repo}/{item['path']}"):
                    contain_pom = True
                    break
                else:
                    os.remove(f"repos/{repo}/{item['path']}")
    print(f"{owner}/{repo} pom checking result: ", contain_pom)
    shutil.rmtree(f"repos/{repo}/")
    return contain_pom
    # # Iterate over the file and directory objects in the response
    # for item in data['tree']:
    #     # Retrieve the file path and type
    #     path, type = item['path'], item['type']

    #     # If the item is a file, retrieve the raw content using the 'url' property
    #     if type == 'blob':
    #         file_url = item['url']
    #         file_response = requests.get(file_url)
    #         file_data = file_response.content

    #         # Process the file content as needed
    #         print(f'File: {path}')
    #         #print(file_data)
    #     else:
    #         # Process directories or other items as needed
    #         print(f'Directory: {path}')
# github.com/davidb/scala-maven-plugin

def check_repo_root(owner, repo, access_token, branch="master"):
    # Define the Github Tree API endpoint and repository details
    #api_url = 'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'
    # Make an HTTP GET request to the Github Tree API endpoint
    #headers = {'Authorization': f'token {access_token}'}
    if not os.path.exists(f"repos/{repo}/"):
        os.makedirs(f"repos/{repo}/")
        #print(f"./{repo}/ created")

    git = Github(access_token)
    try:
        git_repo = git.get_repo(f"{owner}/{repo}")
    except GithubException as e:
        if e.status == 404:
            print("non")
        else:
            print("error")
        shutil.rmtree(f"repos/{repo}/")
        return False
    
    sha = get_sha_for_tag(git_repo, branch)
    # Parse the response data as JSON
    contain_pom = False
    if sha is not None:
        contents = git_repo.get_dir_contents(".", ref=sha)
        for content in contents:
            if content.type == "file" and content.path == "pom.xml":
                download_file(git_repo, sha, repo, content.path)
                if check_log4j(f"repos/{repo}/{content.path}") or check_string_in_file(f"repos/{repo}/{content.path}"):
                    contain_pom = True
                    break

    shutil.rmtree(f"repos/{repo}/")
    print(f"{owner}/{repo} pom checking result: ", contain_pom)
    return contain_pom

#check_repo("davidb", "scala-maven-plugin")