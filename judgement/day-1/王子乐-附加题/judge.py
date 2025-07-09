#!/usr/bin/env python3
"""
Day-1 作业自动批改脚本
用于检查文件完整性并使用大模型API进行评分
支持并行处理多个学生的作业
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import openai
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HomeworkGrader:
    def __init__(self, submission_dir: str, api_key: str, model: str = "deepseek-chat", 
                 max_workers: int = 5, rate_limit: int = 10):
        """
        初始化批改器
        
        Args:
            submission_dir: 提交目录路径
            api_key: OpenAI API密钥
            model: 使用的模型名称
            max_workers: 最大并行工作线程数
            rate_limit: 每分钟最大请求数
        """
        self.submission_dir = Path(submission_dir)
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 速率限制相关
        self.request_times = Queue()
        self.rate_lock = threading.Lock()
        
        # 必需的文件列表
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
            "file_completeness": 20,  # 文件完整性
            "readme_quality": 15,     # README质量
            "code_quality": 25,       # 代码质量
            "log_correctness": 25,    # 日志正确性
            "environment_setup": 15   # 环境配置
        }

    def wait_for_rate_limit(self):
        """
        实现速率限制，确保不超过每分钟的请求限制
        """
        with self.rate_lock:
            current_time = time.time()
            
            # 清理60秒前的请求记录
            while not self.request_times.empty():
                if current_time - self.request_times.queue[0] > 60:
                    self.request_times.get()
                else:
                    break
            
            # 如果请求数已达到限制，等待
            if self.request_times.qsize() >= self.rate_limit:
                sleep_time = 60 - (current_time - self.request_times.queue[0])
                if sleep_time > 0:
                    logger.info(f"达到速率限制，等待 {sleep_time:.2f} 秒...")
                    time.sleep(sleep_time)
                    # 重新清理过期请求
                    current_time = time.time()
                    while not self.request_times.empty():
                        if current_time - self.request_times.queue[0] > 60:
                            self.request_times.get()
                        else:
                            break
            
            # 记录当前请求时间
            self.request_times.put(current_time)

    def check_file_completeness(self, student_dir: Path) -> Tuple[bool, List[str]]:
        """
        检查学生目录中文件的完整性
        
        Args:
            student_dir: 学生目录路径
            
        Returns:
            (是否完整, 缺失文件列表)
        """
        missing_files = []
        
        for required_file in self.required_files:
            file_path = student_dir / required_file
            if not file_path.exists():
                missing_files.append(required_file)
        
        return len(missing_files) == 0, missing_files

    def read_file_content(self, file_path: Path) -> str:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容字符串
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
           # 如果 UTF-8 解码失败，尝试 UTF-16
            try:
                with open(file_path, 'r', encoding='utf-16') as f:
                    return f.read()
            except UnicodeDecodeError:
                # 如果 UTF-16 也失败，再尝试 GBK
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        return f.read()
                except:
                    return f"[无法读取文件内容: {file_path}]"
        except Exception as e:
            return f"[读取文件出错: {e}]"

    def prepare_grading_prompt(self, student_name: str, file_contents: Dict[str, str]) -> str:
        """
        准备批改提示词
        
        Args:
            student_name: 学生姓名
            file_contents: 文件内容字典
            
        Returns:
            批改提示词
        """
        prompt = f"""
作为一名专业的编程作业批改老师，请对学生 {student_name} 的 Day-1 作业进行评分。

## 作业要求回顾
1. **飞书自动化使用**: 修改 doc_viewer.py 查询包含"刘智耿"的条目，输出到 hw1.log
2. **Linux环境使用**: 安装miniconda，创建Python环境，安装PyTorch/Transformers/vllm，运行检查脚本
3. **文件提交**: 提交README.md、代码文件和日志文件

## 评分标准 (总分100分)
- **文件完整性** (20分): 是否提交了所有必需文件
- **README质量** (15分): 自我介绍是否完整、清晰
- **代码质量** (25分): doc_viewer.py的修改是否正确、代码风格
- **日志正确性** (25分): 各个日志文件是否包含预期内容
- **环境配置** (15分): 环境配置是否正确（通过hw2_2.log和hw2_3.log判断）

## 学生提交的文件内容

### README.md
```
{file_contents.get('README.md', '[文件不存在]')}
```

### day-1/README.md
```
{file_contents.get('day-1/README.md', '[文件不存在]')}
```

### day-1/doc_viewer.py
```python
{file_contents.get('day-1/doc_viewer.py', '[文件不存在]')}
```

### day-1/hw1.log
```
{file_contents.get('day-1/hw1.log', '[文件不存在]')}
```

### day-1/hw2_1.log
```
{file_contents.get('day-1/hw2_1.log', '[文件不存在]')}
```

### day-1/hw2_2.log
```
{file_contents.get('day-1/hw2_2.log', '[文件不存在]')}
```

### day-1/hw2_3.log
```
{file_contents.get('day-1/hw2_3.log', '[文件不存在]')}
```

## 评分要求
请按照以下格式返回评分结果：

```json
{{
    "student_name": "{student_name}",
    "total_score": 总分,
    "detailed_scores": {{
        "file_completeness": 分数,
        "readme_quality": 分数,
        "code_quality": 分数,
        "log_correctness": 分数,
        "environment_setup": 分数
    }},
    "feedback": {{
        "strengths": ["优点1", "优点2", ...],
        "improvements": ["改进建议1", "改进建议2", ...],
        "specific_issues": ["具体问题1", "具体问题2", ...]
    }},
    "comments": "总体评价和建议"
}}
```

## 具体评分细则
1. **文件完整性**: 每缺失一个文件扣3分
2. **README质量**:
   - README.md为自我介绍，检查是否包含姓名及简单介绍
   - day-1/README.md为小结，检查是否简单总结本日工作成果
3. **代码质量**: 检查doc_viewer.py是否正确修改查询条件
4. **日志正确性**: 
   - hw1.log应包含查询"刘智耿"的恰好两条结果
   - hw2_1.log应包含nvidia-smi的输出
   - hw2_2.log应包含环境信息（conda envs, python路径, PyTorch版本等）
   - hw2_3.log应包含vLLM的测试结果
5. **环境配置**: 通过hw2_2.log和hw2_3.log判断环境是否正确配置，存在影响不大的warning不扣除分数

请严格按照JSON格式返回，不要包含其他内容。
"""
        return prompt

    def grade_homework(self, student_name: str, file_contents: Dict[str, str]) -> Dict:
        """
        使用大模型评分作业
        
        Args:
            student_name: 学生姓名
            file_contents: 文件内容字典
            
        Returns:
            评分结果字典
        """
        try:
            # 应用速率限制
            self.wait_for_rate_limit()
            
            prompt = self.prepare_grading_prompt(student_name, file_contents)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一名专业的编程作业批改老师，请严格按照要求对作业进行评分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 尝试解析JSON结果
            try:
                # 提取JSON部分
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                json_str = result_text[json_start:json_end]
                
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                logger.error(f"无法解析AI返回的JSON: {result_text}")
                return {
                    "student_name": student_name,
                    "total_score": 0,
                    "detailed_scores": {k: 0 for k in self.grading_criteria.keys()},
                    "feedback": {
                        "strengths": [],
                        "improvements": ["AI评分失败，请人工检查"],
                        "specific_issues": ["JSON解析错误"]
                    },
                    "comments": f"AI评分失败，原始输出: {result_text}"
                }
                
        except Exception as e:
            logger.error(f"评分过程中出错: {e}")
            return {
                "student_name": student_name,
                "total_score": 0,
                "detailed_scores": {k: 0 for k in self.grading_criteria.keys()},
                "feedback": {
                    "strengths": [],
                    "improvements": ["评分失败，请人工检查"],
                    "specific_issues": [f"API调用错误: {str(e)}"]
                },
                "comments": f"评分失败: {str(e)}"
            }

    def process_student(self, student_dir: Path) -> Dict:
        """
        处理单个学生的作业
        
        Args:
            student_dir: 学生目录路径
            
        Returns:
            处理结果字典
        """
        student_name = student_dir.name
        thread_id = threading.current_thread().ident
        logger.info(f"[线程 {thread_id}] 正在处理学生: {student_name}")
        
        # 检查文件完整性
        is_complete, missing_files = self.check_file_completeness(student_dir)
        
        if not is_complete:
            logger.warning(f"[线程 {thread_id}] 学生 {student_name} 缺失文件: {missing_files}")
        
        # 读取所有文件内容
        file_contents = {}
        for required_file in self.required_files:
            file_path = student_dir / required_file
            if file_path.exists():
                file_contents[required_file] = self.read_file_content(file_path)
            else:
                file_contents[required_file] = '[文件不存在]'

        # 将缺失文件信息传递给AI
        if missing_files:
            file_contents['_missing_files'] = missing_files
        
        # 使用AI评分
        result = self.grade_homework(student_name, file_contents)
        
        logger.info(f"[线程 {thread_id}] 学生 {student_name} 评分完成: {result['total_score']}/100")
        return result

    def grade_all_students_parallel(self) -> List[Dict]:
        """
        并行批改所有学生的作业
        
        Returns:
            所有学生的评分结果列表
        """
        results = []
        
        if not self.submission_dir.exists():
            logger.error(f"提交目录不存在: {self.submission_dir}")
            return results
        
        # 收集所有学生目录
        student_dirs = [
            student_dir for student_dir in self.submission_dir.iterdir()
            if student_dir.is_dir() and not student_dir.name.startswith('.')
        ]
        
        logger.info(f"发现 {len(student_dirs)} 个学生目录，开始并行处理...")
        logger.info(f"使用 {self.max_workers} 个工作线程，速率限制: {self.rate_limit} 请求/分钟")
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_student = {
                executor.submit(self.process_student, student_dir): student_dir.name
                for student_dir in student_dirs
            }
            
            # 处理完成的任务
            completed_count = 0
            for future in as_completed(future_to_student):
                student_name = future_to_student[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    logger.info(f"进度: {completed_count}/{len(student_dirs)} 完成")
                except Exception as e:
                    logger.error(f"处理学生 {student_name} 时出错: {e}")
                    results.append({
                        "student_name": student_name,
                        "total_score": 0,
                        "detailed_scores": {k: 0 for k in self.grading_criteria.keys()},
                        "feedback": {
                            "strengths": [],
                            "improvements": ["处理失败，请人工检查"],
                            "specific_issues": [f"处理错误: {str(e)}"]
                        },
                        "comments": f"处理失败: {str(e)}"
                    })
        
        logger.info(f"并行处理完成，共处理 {len(results)} 个学生")
        return results

    def grade_all_students(self) -> List[Dict]:
        """
        批改所有学生的作业（保持原有接口，内部使用并行处理）
        
        Returns:
            所有学生的评分结果列表
        """
        return self.grade_all_students_parallel()

    def grade_all_students_sequential(self) -> List[Dict]:
        """
        顺序批改所有学生的作业（原有方法）
        
        Returns:
            所有学生的评分结果列表
        """
        results = []
        
        if not self.submission_dir.exists():
            logger.error(f"提交目录不存在: {self.submission_dir}")
            return results
        
        # 遍历所有学生目录
        student_dirs = [
            student_dir for student_dir in self.submission_dir.iterdir()
            if student_dir.is_dir() and not student_dir.name.startswith('.')
        ]
        
        logger.info(f"发现 {len(student_dirs)} 个学生目录，开始顺序处理...")
        
        for i, student_dir in enumerate(student_dirs, 1):
            try:
                result = self.process_student(student_dir)
                results.append(result)
                logger.info(f"进度: {i}/{len(student_dirs)} 完成")
            except Exception as e:
                logger.error(f"处理学生 {student_dir.name} 时出错: {e}")
                results.append({
                    "student_name": student_dir.name,
                    "total_score": 0,
                    "detailed_scores": {k: 0 for k in self.grading_criteria.keys()},
                    "feedback": {
                        "strengths": [],
                        "improvements": ["处理失败，请人工检查"],
                        "specific_issues": [f"处理错误: {str(e)}"]
                    },
                    "comments": f"处理失败: {str(e)}"
                })
        
        return results

    def generate_report(self, results: List[Dict]) -> str:
        """
        生成批改报告
        
        Args:
            results: 所有学生的评分结果
            
        Returns:
            报告内容
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Day-1 作业批改报告
生成时间: {timestamp}

## 总体统计
- 总提交人数: {len(results)}
- 平均分: {sum(r['total_score'] for r in results) / len(results) if results else 0:.2f}
- 最高分: {max(r['total_score'] for r in results) if results else 0}
- 最低分: {min(r['total_score'] for r in results) if results else 0}

## 各项得分统计
"""
        
        # 计算各项平均分
        if results:
            for criterion in self.grading_criteria.keys():
                avg_score = sum(r['detailed_scores'].get(criterion, 0) for r in results) / len(results)
                report += f"- {criterion}: {avg_score:.2f}/{self.grading_criteria[criterion]}\n"
        
        report += "\n## 个人详细评分\n\n"
        
        # 按分数排序
        sorted_results = sorted(results, key=lambda x: x['total_score'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            report += f"### {i}. {result['student_name']} - {result['total_score']}/100\n\n"
            
            # 详细分数
            report += "**详细分数:**\n"
            for criterion, score in result['detailed_scores'].items():
                max_score = self.grading_criteria[criterion]
                report += f"- {criterion}: {score}/{max_score}\n"
            
            # 反馈
            if result['feedback']['strengths']:
                report += "\n**优点:**\n"
                for strength in result['feedback']['strengths']:
                    report += f"- {strength}\n"
            
            if result['feedback']['improvements']:
                report += "\n**改进建议:**\n"
                for improvement in result['feedback']['improvements']:
                    report += f"- {improvement}\n"
            
            if result['feedback']['specific_issues']:
                report += "\n**具体问题:**\n"
                for issue in result['feedback']['specific_issues']:
                    report += f"- {issue}\n"
            
            report += f"\n**总体评价:** {result['comments']}\n\n"
            report += "---\n\n"
        
        return report

    def save_results(self, results: List[Dict], output_dir: str = "grading_results"):
        """
        保存批改结果
        
        Args:
            results: 评分结果列表
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_file = output_path / f"grading_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成并保存报告
        report = self.generate_report(results)
        report_file = output_path / f"grading_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"结果已保存到 {output_path}")
        logger.info(f"JSON结果: {json_file}")
        logger.info(f"报告文件: {report_file}")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Day-1 作业自动批改脚本')
    parser.add_argument('--submission-dir', default='../../../submission', help='提交目录路径')
    parser.add_argument('--api-key', default='sk-930216e611124f928f1903596efc6723', help='API密钥')
    parser.add_argument('--model', default='deepseek-chat', help='使用的模型名称')
    parser.add_argument('--output-dir', default='grading_results', help='输出目录')
    parser.add_argument('--max-workers', type=int, default=5, help='最大并行工作线程数')
    parser.add_argument('--rate-limit', type=int, default=20, help='每分钟最大请求数')
    parser.add_argument('--sequential', action='store_true', help='使用顺序处理而非并行处理')
    
    args = parser.parse_args()
    
    # 创建批改器
    grader = HomeworkGrader(
        submission_dir=args.submission_dir,
        api_key=args.api_key,
        model=args.model,
        max_workers=args.max_workers,
        rate_limit=args.rate_limit
    )
    
    # 批改所有作业
    start_time = time.time()
    logger.info("开始批改作业...")
    
    if args.sequential:
        logger.info("使用顺序处理模式...")
        results = grader.grade_all_students_sequential()
    else:
        logger.info("使用并行处理模式...")
        results = grader.grade_all_students_parallel()
    
    end_time = time.time()
    logger.info(f"批改耗时: {end_time - start_time:.2f} 秒")
    
    # 保存结果
    grader.save_results(results, args.output_dir)
    
    logger.info("批改完成!")


if __name__ == "__main__":
    main()