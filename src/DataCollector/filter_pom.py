import json
import subprocess
from check_pom import check_repo_root
from tqdm import tqdm

with open("results.json", encoding='latin1') as rf:
    repos = json.load(rf)


Key = ""

repos = repos['items']
#check_repo_root("nysenate", "openlegislation", "dev")
end_point = len(repos)

with open("result.txt", "a") as f:
    for i in range(17, end_point):
        repo_item = repos[i]
        branch = repo_item['defaultBranch']
        owner, repo = repo_item['name'].split('/')
        print(f"\n{i}-{end_point}/{len(repos)} repo: {owner} {repo} {branch}\n")
        if check_repo_root(owner, repo, key, branch):
            f.write(f"{i} {owner} {repo} {branch}\n")
            f.flush()
