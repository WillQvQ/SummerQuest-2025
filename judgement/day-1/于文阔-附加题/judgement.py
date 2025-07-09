# 本代码的框架由AI生成

#!/usr/bin/env python3
"""
暑期集训Day-1作业自动批改系统
用于检查学生提交的作业文件并进行自动评分
"""

import os
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import openai
from openai import OpenAI

class HomeworkGrader:
    def __init__(self, openai_api_key: str, submission_dir: str = "submission"):
        """
        初始化批改系统
        
        Args:
            openai_api_key: OpenAI API密钥
            submission_dir: 提交目录路径
        """
        self.client = OpenAI(api_key=openai_api_key, base_url="https://api.deepseek.com")
        self.submission_dir = Path(submission_dir)
        self.required_files = [
            "README.md",
            "day-1/README.md",
            "day-1/doc_viewer.py",
            "day-1/hw1.log",
            "day-1/hw2_1.log",
            "day-1/hw2_2.log",
            "day-1/hw2_3.log"
        ]
        
        # 评分标准
        self.grading_criteria = {
            "README.md": {
                "weight": 10,
                "requirements": "个人介绍，标题是自己的名字,直接给100分"
            },
            "day-1/README.md": {
                "weight": 10,
                "requirements": "个人小结,直接给100分"
            },
            "day-1/doc_viewer.py": {
                "weight": 20,
                "requirements": """修改后的代码，查询包含'刘智耿'的条目，如果代码可以运行就给100分。
                """
            },
            "day-1/hw1.log": {
                "weight": 15,
                "requirements": """
                飞书自动化查询结果输出,如果筛选结果如下，则给100分，否则给0分，
                筛选记录 1:
                主讲: 刘智耿
                助教: 怀天宇, 方世成, 宋悦荣
                日期: Day-5（7.11）
                课程: 常见框架介绍( llamafactory, verl, vllm)

                筛选记录 2:
                主讲: 郑逸宁
                助教: 陈敬麒, 刘智耿
                日期: Day-1（7.7）
                课程: Slurm 使用方法（组内新生用 slurm）
            """
            },
            "day-1/hw2_1.log": {
                "weight": 15,
                "requirements": "nvidia-smi输出结果，包含显卡信息(100分)"
            },
            "day-1/hw2_2.log": {
                "weight": 15,
                "requirements": "env_checker.py运行结果，包含环境信息(34分),对普通对话模式的测试(33分),对思维链模式的测试(33分)"
            },
            "day-1/hw2_3.log": {
                "weight": 15,
                "requirements": "vllm_checker.py运行结果，包含环境信息(34分),对普通对话模式的测试(33分),对思维链模式的测试(33分)，如果没有和vllm相关的内容，则给0分"
            }
        }

    def get_student_directories(self) -> List[Path]:
        """获取所有学生目录"""
        if not self.submission_dir.exists():
            print(f"提交目录 {self.submission_dir} 不存在")
            return []
        
        student_dirs = []
        for item in self.submission_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                student_dirs.append(item)
        
        return student_dirs

    def check_file_existence(self, student_dir: Path) -> Dict[str, bool]:
        """检查学生目录中必需文件是否存在"""
        file_status = {}
        for file_path in self.required_files:
            full_path = student_dir / file_path
            file_status[file_path] = full_path.exists()
        return file_status

    def read_file_content(self, file_path: Path) -> str:
        """安全地读取文件内容"""
        if not file_path.exists():
            return ""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except:
                return f"无法读取文件 {file_path}"
        except Exception as e:
            return f"读取文件时出错: {str(e)}"

    def evaluate_content_with_llm(self, file_path: str, content: str, requirements: str) -> Tuple[int, str]:
        """使用大模型评估文件内容质量"""
        if not content.strip():
            return 0, "文件为空"
        
        prompt = f"""
请作为一个作业批改老师，评估以下学生提交的文件内容：

文件路径: {file_path}
要求: {requirements} (请严格按照这个要求进行评估)

文件内容:
{content}

请给出分数和详细的评价理由。
格式：分数: XX
理由: 详细解释
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的作业批改助手，需要给出客观公正的评分。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            
            # 解析分数
            score_match = re.search(r'分数[:：]\s*(\d+)', result)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(100, score))  # 确保分数在0-100之间
            else:
                score = 50  # 默认分数
            
            return score, result
            
        except Exception as e:
            return 50, f"AI评估出错: {str(e)}"

    def grade_student(self, student_dir: Path) -> Dict:
        """为单个学生进行批改"""
        student_name = student_dir.name
        print(f"正在批改学生: {student_name}")
        
        result = {
            "student_name": student_name,
            "timestamp": datetime.now().isoformat(),
            "file_checks": {},
            "content_evaluations": {},
            "total_score": 0,
            "max_score": 100,
            "missing_files": [],
            "comments": []
        }
        
        # 检查文件存在性
        file_status = self.check_file_existence(student_dir)
        result["file_checks"] = file_status
        
        # 统计缺失文件
        for file_path, exists in file_status.items():
            if not exists:
                result["missing_files"].append(file_path)
        
        # 评估每个文件的内容
        total_weighted_score = 0
        total_weight = 0
        
        for file_path in self.required_files:
            if file_status.get(file_path, False):
                full_path = student_dir / file_path
                content = self.read_file_content(full_path)
                
                requirements = self.grading_criteria[file_path]["requirements"]
                weight = self.grading_criteria[file_path]["weight"]
                
                # 使用AI评估内容
                score, evaluation = self.evaluate_content_with_llm(file_path, content, requirements)
                
                result["content_evaluations"][file_path] = {
                    "score": score,
                    "evaluation": evaluation,
                    "weight": weight
                }
                
                total_weighted_score += score * weight / 100
                total_weight += weight
            else:
                # 文件不存在，该项得0分
                weight = self.grading_criteria[file_path]["weight"]
                result["content_evaluations"][file_path] = {
                    "score": 0,
                    "evaluation": "文件不存在",
                    "weight": weight
                }
                total_weight += weight
        
        # 计算总分
        if total_weight > 0:
            result["total_score"] = round(total_weighted_score / total_weight * 100, 2)
        
        # 添加评语
        if len(result["missing_files"]) > 0:
            result["comments"].append(f"缺失文件: {', '.join(result['missing_files'])}")
        
        if result["total_score"] >= 90:
            result["comments"].append("优秀！作业完成质量很高。")
        elif result["total_score"] >= 80:
            result["comments"].append("良好，作业基本完成，还有改进空间。")
        elif result["total_score"] >= 70:
            result["comments"].append("及格，但需要更加认真完成作业。")
        else:
            result["comments"].append("需要重新提交，作业完成度不足。")
        
        return result

    def grade_all_students(self) -> Dict:
        """批改所有学生的作业"""
        print("开始批改所有学生作业...")
        
        student_dirs = self.get_student_directories()
        if not student_dirs:
            print("没有找到学生提交目录")
            return {}
        
        all_results = {
            "grading_time": datetime.now().isoformat(),
            "total_students": len(student_dirs),
            "students": {},
            "statistics": {}
        }
        
        scores = []
        for student_dir in student_dirs:
            try:
                result = self.grade_student(student_dir)
                all_results["students"][result["student_name"]] = result
                scores.append(result["total_score"])
                print(f"学生 {result['student_name']} 批改完成，得分: {result['total_score']}")
            except Exception as e:
                print(f"批改学生 {student_dir.name} 时出错: {str(e)}")
                all_results["students"][student_dir.name] = {
                    "error": str(e),
                    "total_score": 0
                }
        
        # 计算统计信息
        if scores:
            all_results["statistics"] = {
                "average_score": round(sum(scores) / len(scores), 2),
                "max_score": max(scores),
                "min_score": min(scores),
                "pass_rate": len([s for s in scores if s >= 70]) / len(scores) * 100
            }
        
        return all_results

    def generate_report(self, results: Dict, output_file: str = "grading_report.json"):
        """生成批改报告"""
        # 保存详细结果到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成简要报告
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("暑期集训Day-1作业批改报告")
        report_lines.append("=" * 80)
        report_lines.append(f"批改时间: {results['grading_time']}")
        report_lines.append(f"总学生数: {results['total_students']}")
        # report_lines.append()
        
        if "statistics" in results:
            stats = results["statistics"]
            report_lines.append("统计信息:")
            report_lines.append(f"平均分: {stats['average_score']}")
            report_lines.append(f"最高分: {stats['max_score']}")
            report_lines.append(f"最低分: {stats['min_score']}")
            report_lines.append(f"及格率: {stats['pass_rate']:.1f}%")
            # report_lines.append()
        
        report_lines.append("学生成绩详情:")
        report_lines.append("-" * 80)
        
        for student_name, result in results["students"].items():
            if "error" in result:
                report_lines.append(f"{student_name}: 批改出错 - {result['error']}")
            else:
                score = result["total_score"]
                missing = len(result["missing_files"])
                report_lines.append(f"{student_name}: {score}分 (缺失文件: {missing}个)")
                
                if result["comments"]:
                    for comment in result["comments"]:
                        report_lines.append(f"  评语: {comment}")
        
        report_lines.append("=" * 80)
        
        # 保存简要报告
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"详细报告已保存至: {output_file}")
        print(f"简要报告已保存至: {summary_file}")
        
        return '\n'.join(report_lines)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="暑期集训Day-1作业自动批改系统")
    parser.add_argument("--api-key", required=True, help="OpenAI API密钥")
    parser.add_argument("--submission-dir", default="submission", help="提交目录路径")
    parser.add_argument("--output", default="grading_report.json", help="输出报告文件名")
    
    args = parser.parse_args()
    
    # 初始化批改系统
    grader = HomeworkGrader(args.api_key, args.submission_dir)
    
    # 批改所有学生作业
    results = grader.grade_all_students()
    
    # 生成报告
    summary = grader.generate_report(results, args.output)
    print("\n" + summary)

if __name__ == "__main__":
    main()

#
#D:/anaconda3/python.exe e:/nlp/SummerQuest-2025/submission/于文阔/day-1/judgement.py --api-key sk-ce297dad0b374efe942c1ff951b758dc --submission-dir ./submission --output ./output.txt