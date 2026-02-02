import json
import requests
from tqdm import tqdm
import random
import time

# 从配置文件导入
from config import GITHUB_TOKENS, OWNER, REPO, ISSUES_FILE, COMMIT_ISSUE_MAP_FILE

# 随机获取一个 Token
def get_random_token():
    return random.choice(GITHUB_TOKENS)

# 带异常处理的请求
# 带异常处理的请求
def make_request_with_retry(url, headers, retries=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 401:  # Token 无效
                print(f"Invalid token, switching to another token... Attempt {attempt + 1}")
                headers["Authorization"] = f"token {get_random_token()}"
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:  # 最后一次失败时暂停
                print("Max retries exceeded. Pausing for 6 hours...")
                time.sleep(6 * 60 * 60)  # 暂停 6 小时
                print("Resuming after 6 hours...")
            else:
                headers["Authorization"] = f"token {get_random_token()}"
    raise Exception("Max retries exceeded. Unable to complete the request.")


# 读取本地 issues 数据
def load_issues_from_file(issues_file):
    try:
        with open(issues_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"{issues_file} not found. Please run the issue fetching script first.")
        return []

# 获取 issue 的 IssueEvents 并筛选 referenced 类型的事件
def get_issue_events(owner, repo, issue_number, headers):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/events"
    events = []
    response = make_request_with_retry(url, headers)
    if response:
        data = response.json()
        for event in data:
            if event.get("event") == "referenced" and "commit_id" in event:
                events.append({"commit_id": event["commit_id"], "issue_number": issue_number})
    return events

# 保存映射到本地文件
def save_to_json_file(data, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# 构建 commit 到 issues 的映射
def build_commit_issue_mapping(issues, owner, repo, output_file):
    # 如果文件已存在，加载已有数据，避免重复处理
    try:
        with open(output_file, "r", encoding="utf-8") as file:
            commit_issue_map = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        commit_issue_map = {}

    headers = {"Authorization": f"token {get_random_token()}"}

    # 使用 tqdm 为 issues 添加进度条
    for issue in tqdm(issues, desc="Processing issues", unit="issue"):
        issue_number = issue["number"]
        events = get_issue_events(owner, repo, issue_number, headers)

        for event in events:
            commit_id = event["commit_id"]
            if commit_id not in commit_issue_map:
                commit_issue_map[commit_id] = []
            if issue_number not in commit_issue_map[commit_id]:
                commit_issue_map[commit_id].append(issue_number)

        # 每处理一个 issue，就保存一次到 JSON 文件
        save_to_json_file(commit_issue_map, output_file)

    return commit_issue_map

# 主函数
if __name__ == "__main__":
    # 加载本地 issues 数据
    issues = load_issues_from_file(ISSUES_FILE)
    if issues:
        commit_issue_mapping = build_commit_issue_mapping(issues, OWNER, REPO, COMMIT_ISSUE_MAP_FILE)
        print(f"Commit to Issue Mapping saved to {COMMIT_ISSUE_MAP_FILE}")
    else:
        print("No issues data available. Please fetch issues first.")