import os

REQUIRED_FILES = [
    "README.md",
    "day-1/README.md",
    "day-1/hw1.log",
    "day-1/hw2_1.log",
    "day-1/hw2_2.log",
    "day-1/hw2_3.log"
]

def check_files(base_path):
    missing = []
    for f in REQUIRED_FILES:
        if not os.path.exists(os.path.join(base_path, f)):
            missing.append(f)
    return missing

def check_hw1_log(log_path):
    try:
        with open(log_path, encoding="utf-8") as f:
            content = f.read()
        return "刘智耿" in content and ("主讲" in content or "助教" in content)
    except:
        return False

def check_hw2_1(log_path):
    try:
        with open(log_path) as f:
            content = f.read()
        keywords = ["NVIDIA", "Driver Version", "CUDA Version"]
        return all(k in content for k in keywords)
    except:
        return False

def check_hw2_2(log_path):
    try:
        with open(log_path, encoding="utf-8") as f:
            content = f.read()
        return (
            "PyTorch 版本:" in content and
            "Transformers 版本:" in content and
            "CUDA 是否可用: True" in content and
            "可用 GPU 数量: 1" in content
        )
    except Exception:
        return False


def check_hw2_3(log_path):
    try:
        with open(log_path) as f:
            content = f.read()
        return "vllm" in content.lower()
    except:
        return False

def grade_student(user_dir):
    score = 0
    total = 100
    student_name = os.path.basename(user_dir)
    print(f"\n🎯 学生：{student_name}")
    print("=" * 40)

    # 1. 文件检查
    missing = check_files(user_dir)
    if not missing:
        print("✅ 文件检查通过 (20分)")
        score += 20
    else:
        print(f"❌ 缺少文件: {', '.join(missing)} (0分)")

    # 2. hw1.log 检查
    hw1_ok = check_hw1_log(os.path.join(user_dir, "day-1/hw1.log"))
    print(f"{'✅' if hw1_ok else '❌'} hw1.log 查找刘智耿 {'(20分)' if hw1_ok else '(0分)'}")
    if hw1_ok: score += 20

    # 3. hw2_1.log 检查
    hw2_1_ok = check_hw2_1(os.path.join(user_dir, "day-1/hw2_1.log"))
    print(f"{'✅' if hw2_1_ok else '❌'} hw2_1.log GPU 信息检查 {'(20分)' if hw2_1_ok else '(0分)'}")
    if hw2_1_ok: score += 20

    # 4. hw2_2.log 检查
    hw2_2_ok = check_hw2_2(os.path.join(user_dir, "day-1/hw2_2.log"))
    print(f"{'✅' if hw2_2_ok else '❌'} hw2_2.log 环境依赖检查 {'(20分)' if hw2_2_ok else '(0分)'}")
    if hw2_2_ok: score += 20

    # 5. hw2_3.log 检查
    hw2_3_ok = check_hw2_3(os.path.join(user_dir, "day-1/hw2_3.log"))
    print(f"{'✅' if hw2_3_ok else '❌'} hw2_3.log VLLM 检查 {'(20分)' if hw2_3_ok else '(0分)'}")
    if hw2_3_ok: score += 20

    # 总分
    print("=" * 40)
    print(f"📊 {student_name} 最终得分: {score} / {total}")
    if score == 100:
        print("🎉 作业通过！满分！")
    elif score >= 60:
        print("👍 作业基本合格")
    else:
        print("⚠️ 作业不合格，请检查问题")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../submission"))
    if not os.path.exists(base_dir):
        print("❌ 未找到 submission 目录")
        return

    # 遍历每个学生的目录
    for student in sorted(os.listdir(base_dir)):
        student_path = os.path.join(base_dir, student)
        if os.path.isdir(student_path):
            grade_student(student_path)

if __name__ == "__main__":
    main()
