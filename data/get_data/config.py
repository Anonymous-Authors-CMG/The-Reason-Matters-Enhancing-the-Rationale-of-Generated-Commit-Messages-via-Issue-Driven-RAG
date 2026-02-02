# ============================================================
# 配置文件 - 在此修改所有参数，然后运行 run_all.py
# ============================================================

# GitHub API Tokens (可以配置多个用于轮询，避免限流)
GITHUB_TOKENS = [
    "xxxxx",
    # "token2",
    # "token3",
]

# 仓库信息
OWNER = "tensorflow"      # 仓库所有者
REPO = "tensorflow"       # 仓库名

# Git 本地仓库路径 (用于获取 commit 详情)
# 请确保已经 clone 了对应的仓库
GIT_REPO_PATH = "./repo_code/tensorflow.git"

# ============================================================
# 中间文件名配置 (一般不需要修改)
# ============================================================
ISSUES_FILE = "1_issues_data.json"                      # 步骤1输出
COMMIT_ISSUE_MAP_FILE = "2_commit_issue_mapping.json"   # 步骤2输出
COMMIT_DETAIL_FILE = "3.0_commit_detail.json"           # 步骤3.0输出
REMOVE_BOT_FILE = "3.1_remove_bot.json"                 # 步骤3.1输出
FILTERED_DIFF_FILE = "3.2_remove_4096_token_diff.json"  # 步骤3.2输出
TOKEN_LENGTH_DIST_FILE = "3.2_token_length_distribution.txt"
FINAL_OUTPUT_FILE = "4_merge_output.json"               # 最终输出

# Bot 排除列表
BOT_EXCLUSION_LIST = [
    "dependabot-preview[bot]",
    "dependabot[bot]",
    "Renovate Bot",
    "nextcloud-android-bot",
    "Gary Bot"
]

