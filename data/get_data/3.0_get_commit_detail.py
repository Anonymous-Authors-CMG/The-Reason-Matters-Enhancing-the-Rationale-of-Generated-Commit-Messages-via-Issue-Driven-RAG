import json
from git import Repo, BadName
from tqdm import tqdm

# 从配置文件导入
from config import COMMIT_ISSUE_MAP_FILE, GIT_REPO_PATH, COMMIT_DETAIL_FILE

# 加载 commit-issue 映射关系
with open(COMMIT_ISSUE_MAP_FILE, 'r') as f:
    commit_issue_mapping = json.load(f)

# 初始化 Git 仓库（注意路径需要指向你的 .git 目录）
repo = Repo(GIT_REPO_PATH)

commit_details = []
valid_commits = 0
invalid_commits = []

# 使用 tqdm 创建进度条
for commit_id in tqdm(commit_issue_mapping.keys(), desc='Processing commits'):
    try:
        commit = repo.commit(commit_id)

        # 获取 diff（对比父提交）
        if commit.parents:
            diff = repo.git.diff(commit.parents[0], commit)
        else:  # 处理初始提交
            diff = repo.git.show(commit.hexsha)

        # 获取提交作者信息
        author_name = commit.author.name
        author_email = commit.author.email

        # 构建数据结构
        commit_details.append({
            "commit id": commit_id,
            "diff": diff,
            "message": commit.message.strip(),
            "author_name": author_name,
            "author_email": author_email
        })
        valid_commits += 1
    except BadName:
        invalid_commits.append(commit_id)
    except Exception as e:
        print(f"Error processing commit {commit_id}: {str(e)}")
        invalid_commits.append(commit_id)

# 保存结果
with open(COMMIT_DETAIL_FILE, 'w') as f:
    json.dump(commit_details, f, indent=2)

# 打印统计信息
print(f"\n处理完成！成功处理 {valid_commits} 个提交")
if invalid_commits:
    print(f"以下 {len(invalid_commits)} 个提交无法找到：")
    print('\n'.join(invalid_commits))