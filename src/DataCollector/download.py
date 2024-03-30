from get_java import download_java

Key = ""

with open("1.txt", "r") as file:
    for line in file:
        repo_list = line.split()
        owner, repo, branch = repo_list[1], repo_list[2], repo_list[3]
        print(f"{repo_list[0]} repo: {owner} {repo} {branch}")
        Done = False
        with open("result1.txt", "r") as f:
            content = f.read()
            if owner in content and repo in content:
                Done = True
        if Done:
            print("Already Done!")
            continue
        cnt1, cnt2 = download_java(owner, repo, key, branch)
        with open("result1.txt", "a") as f:
            f.write(f"{repo_list[0]} {owner}/{repo} downloaded: {cnt1}/{cnt1+cnt2} files\n")