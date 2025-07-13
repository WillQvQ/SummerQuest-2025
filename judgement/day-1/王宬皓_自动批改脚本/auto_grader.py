import os
import re
import subprocess
import json
from pathlib import Path

# 1. 检查文件结构
def check_file_structure(submission_path):
    required_files = [
        "README.md",
        "day-1/README.md",
        "day-1/doc_viewer.py",
        "day-1/hw1.log",
        "day-1/hw2_1.log",
        "day-1/hw2_2.log",
        "day-1/hw2_3.log"
    ]
    missing = []
    for file in required_files:
        if not os.path.exists(os.path.join(submission_path, file)):
            missing.append(file)
    if missing:
        print("❌ 缺少以下文件：", missing)
        return False
    print("✅ 文件结构完整")
    return True

# 2. 检查日志内容
def check_log_content(log_path, pattern):
    with open(log_path, 'r') as f:
        content = f.read()
    if re.search(pattern, content):
        print(f"✅ {log_path.split('/')[-1]} 内容符合要求")
        return True
    else:
        print(f"❌ {log_path.split('/')[-1]} 内容不符合要求")
        return False

# 3. 检查 doc_viewer.py 是否修改正确
def check_doc_viewer(path):
    with open(path, 'r') as f:
        content = f.read()
    if "刘智耿" in content:
        print("✅ doc_viewer.py 修改正确")
        return True
    else:
        print("❌ doc_viewer.py 未正确修改")
        return False

# 4. 检查 conda 环境
def check_conda_env():
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    if "verl" in result.stdout:
        print("✅ conda 环境 verl 存在")
        return True
    else:
        print("❌ 未找到 conda 环境 verl")
        return False

# 5. 检查 PyTorch、Transformers、vLLM 是否安装
def check_packages():
    result = subprocess.run(["pip", "show", "torch"], capture_output=True, text=True)
    if "Name: torch" in result.stdout:
        print("✅ PyTorch 已安装")
    else:
        print("❌ PyTorch 未安装")
    result = subprocess.run(["pip", "show", "transformers"], capture_output=True, text=True)
    if "Name: transformers" in result.stdout:
        print("✅ Transformers 已安装")
    else:
        print("❌ Transformers 未安装")
    result = subprocess.run(["pip", "show", "vLLM"], capture_output=True, text=True)
    if "Name: vLLM" in result.stdout:
        print("✅ vLLM 已安装")
    else:
        print("❌ vLLM 未安装")

# 6. 运行 doc_viewer.py 并检查输出
def run_doc_viewer():
    result = subprocess.run(["python", "doc_viewer.py"], capture_output=True, text=True)
    if "刘智耿" in result.stdout:
        print("✅ doc_viewer.py 输出正确")
        return True
    else:
        print("❌ doc_viewer.py 输出不正确")
        return False

# 主函数
def main():
    submission_path = "../../submission/王宬皓/day-1"
    if not check_file_structure(submission_path):
        return

    # 检查日志文件内容
    check_log_content(f"{submission_path}/day-1/hw1.log", "刘智耿")
    check_log_content(f"{submission_path}/day-1/hw2_1.log", "NVIDIA-SMI")
    check_log_content(f"{submission_path}/day-1/hw2_2.log", "PyTorch 版本")
    check_log_content(f"{submission_path}/day-1/hw2_3.log", "vLLM 版本")

    # 检查 doc_viewer.py
    check_doc_viewer(f"{submission_path}/day-1/doc_viewer.py")

    # 检查 conda 环境和包
    check_conda_env()
    check_packages()

    # 运行测试脚本
    run_doc_viewer()

if __name__ == "__main__":
    main()