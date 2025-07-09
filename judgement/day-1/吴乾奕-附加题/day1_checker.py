import os
import sys
import re

def check_readme(path):
    if not os.path.exists(path):
        return False, "README.md 不存在"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        name_present = re.search(r"#\s*[\u4e00-\u9fa5\w]+", content)
        intro_present = len(content.strip().splitlines()) >= 2
        if not name_present:
            return False, "README.md 缺少姓名标题"
        if not intro_present:
            return False, "README.md 缺少自我介绍"
    return True, "README.md 检查通过"

def check_doc_viewer(path):
    if not os.path.exists(path):
        return False, "doc_viewer.py 不存在"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        if "刘智耿" not in content:
            return False, "doc_viewer.py 未包含查找 '刘智耿' 的逻辑"
    return True, "doc_viewer.py 检查通过"

def check_log_file(path, keywords):
    if not os.path.exists(path):
        return False, f"{os.path.basename(path)} 不存在"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        for kw in keywords:
            if kw not in content:
                return False, f"{os.path.basename(path)} 缺少关键字: {kw}"
    return True, f"{os.path.basename(path)} 检查通过"

def main():
    if len(sys.argv) < 2:
        print("用法: python day1_checker.py [学生目录名]")
        return

    student = sys.argv[1]
    BASE_PATH = os.path.join("submission", student)
    DAY1_PATH = os.path.join(BASE_PATH, "day-1")

    REQUIRED_FILES = {
        "README.md": check_readme,
        "day-1/README.md": check_readme,
        "day-1/doc_viewer.py": check_doc_viewer,
        "day-1/hw1.log": lambda p: check_log_file(p, ["刘智耿"]),
        "day-1/hw2_1.log": lambda p: check_log_file(p, ["NVIDIA", "Driver Version"]),
        "day-1/hw2_2.log": lambda p: check_log_file(p, ["PyTorch", "CUDA", "Transformers"]),
        "day-1/hw2_3.log": lambda p: check_log_file(p, ["vLLM", "初始化", "AI回复"]),
    }

    print(f"\n=== 自动检查：{student} ===")
    results = []
    for rel_path, checker in REQUIRED_FILES.items():
        abs_path = os.path.join(BASE_PATH, rel_path.replace("day-1/", "day-1/"))
        results.append(checker(abs_path))

    for passed, msg in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {msg}")

    if all(p for p, _ in results):
        print(f"\n{student} 作业检查通过！")
    else:
        print(f"\n{student} 作业存在问题，请修改后重新提交。")

if __name__ == "__main__":
    main()

