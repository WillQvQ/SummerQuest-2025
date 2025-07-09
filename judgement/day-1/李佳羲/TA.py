# ==============================================================================
# 第一部分: 配置与常量区
# ==============================================================================
import os 
import sys
import traceback
import time
from zhipuai import ZhipuAI

# API 和模型配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
MODEL_ID = "glm-4"
TEMPERATURE = 0.1

# 文件路径和作业要求配置
PROMPT_TEMPLATE_PATH = "prompts/day-1.md"
REPORTS_OUTPUT_DIR = "reports" # 输出报告
REQUIRED_FILES = [
    'README.md',
    'day-1/README.md',
    'day-1/doc_viewer.py',
    'day-1/hw1.log',
    'day-1/hw2_1.log',
    'day-1/hw2_2.log',
    'day-1/hw2_3.log'
]
# 定义要跳过的学生名单
SKIP_LIST = ["张三", "李四"]

# ==============================================================================
# 第二部分: 主逻辑类 (这部分基本不变)
# ==============================================================================
class TAParser:
    def __init__(self, client):
        self.client = client

    def grade_single_student(self, student_path: str) -> str:
        """核心批改逻辑，只负责处理单个学生"""
        student_name = os.path.basename(student_path)
        print(f"\n{'='*20} 开始批改: {student_name} {'='*20}")
        
        # 1. 构建 Prompt
        try:
            full_prompt = self._build_prompt(student_path, student_name)
            if full_prompt is None: # 如果构建失败（比如缺少文件），则返回错误报告
                return f"为 {student_name} 构建批改指令时失败，可能缺少必要文件，请检查日志。"
            print("  [成功] 构建批改指令")
        except Exception as e:
            return f"为 {student_name} 构建批改指令时发生严重错误: {e}"
        
        # 2. 调用 API
        print(f"  [Info] 正在调用智谱AI大模型进行批改，请稍候...")
        report = self._call_zhipu_api(full_prompt, student_name)
        
        print(f"  [成功] 已完成对 {student_name} 的批改")
        return report

    def _build_prompt(self, student_path: str, student_name: str) -> str or None:
        with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        submission_data = {'student_name': student_name}
        print(f"  [Info] 开始读取 {student_name} 提交的文件...")
        
        all_files_found = True
        for file_path_relative in REQUIRED_FILES:
            full_file_path = os.path.join(student_path, file_path_relative)
            key_name = file_path_relative.replace('/', '_').replace('.', '_').replace('-', '_') + "_content"
            try:
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    submission_data[key_name] = f.read()
            except FileNotFoundError:
                print(f"  [失败] 文件未找到: {full_file_path}")
                submission_data[key_name] = f"错误：此文件 '{file_path_relative}' 未找到或未提交。"
                all_files_found = False # 标记有文件缺失
            except Exception as e:
                error_msg = f"读取文件 {file_path_relative} 时发生错误: {e}"
                print(f"  [失败] {error_msg}")
                submission_data[key_name] = error_msg
                all_files_found = False
        

        return template_content.format(**submission_data)

    def _call_zhipu_api(self, prompt: str, student_name: str) -> str:
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )
            end_time = time.time()
            print(f"  [Info] API 响应成功，耗时: {end_time - start_time:.2f} 秒")
            return response.choices[0].message.content
        except Exception as e:
            end_time = time.time()
            return f"为 {student_name} 调用智谱AI API 时发生错误 (已等待 {end_time - start_time:.2f} 秒): {e}"

# ==============================================================================
# 第三部分: 全局辅助函数
# ==============================================================================
def save_report(student_name: str, report_content: str):
    """将批改报告保存到文件"""
    if not os.path.exists(REPORTS_OUTPUT_DIR):
        os.makedirs(REPORTS_OUTPUT_DIR)
        print(f"创建报告目录: {REPORTS_OUTPUT_DIR}")
        
    report_filename = f"{student_name}_report.md"
    report_filepath = os.path.join(REPORTS_OUTPUT_DIR, report_filename)
    
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"  [成功] 报告已保存至: {report_filepath}")

# ==============================================================================
# 第四部分: 程序主入口 (全新重构)
# ==============================================================================
if __name__ == "__main__":
    # 1. 检查和获取总提交目录
    if len(sys.argv) < 2:
        print("错误: 参数不足！")
        print("用法: python TA.py <path_to_submissions_directory>")
        print("示例: python TA.py ../../submission")
        sys.exit(1)
    
    submissions_dir = sys.argv[1]
    if not os.path.isdir(submissions_dir):
        print(f"错误: 目录不存在 -> {submissions_dir}")
        sys.exit(1)

    # 2. 初始化 API 客户端
    if not ZHIPU_API_KEY:
        print("错误: 找不到环境变量 ZHIPU_API_KEY。请先设置。")
        sys.exit(1)
    
    try:
        zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)
        print("智谱AI API 客户端初始化成功。")
    except Exception as e:
        print(f"初始化智谱AI客户端时发生错误: {e}")
        sys.exit(1)
        
    # 3. 创建批改器实例
    grader = TAParser(zhipu_client)

    # 4. 遍历所有学生目录并执行批改
    print(f"\n开始扫描总提交目录: {submissions_dir}")
    
    # 获取所有学生文件夹名称
    all_students = sorted([d for d in os.listdir(submissions_dir) if os.path.isdir(os.path.join(submissions_dir, d))])
    
    total_students = len(all_students)
    graded_count = 0
    
    for student_name in all_students:
        if student_name in SKIP_LIST:
            print(f"\n{'='*20} 跳过: {student_name} (在忽略名单中) {'='*20}")
            continue

        student_path = os.path.join(submissions_dir, student_name)
        
        # 执行单个学生的批改
        final_report = grader.grade_single_student(student_path)
        
        # 保存报告
        save_report(student_name, final_report)
        graded_count += 1
        
    print(f"\n{'='*25} 全部批改完成 {'='*25}")
    print(f"总计扫描到 {total_students} 个学生目录。")
    print(f"成功批改 {graded_count} 人。")
    print(f"跳过 {len(SKIP_LIST)} 人: {', '.join(SKIP_LIST)}")
    print(f"所有报告已保存在 '{REPORTS_OUTPUT_DIR}' 目录下。")