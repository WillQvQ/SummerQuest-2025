import asyncio
import json
import os
import re
import aiohttp
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class FilteredPaper:
    """筛选后的论文信息"""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    url: str
    relevance_score: float
    relevance_explanation: str
    source_paper: str  # 来源论文的arxiv_id
    relationship: str  # 'citing' 或 'referenced'

class ArxivCitationFilter:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化筛选器
        
        Args:
            config: 配置字典，包含:
                - deepseek_api_key: DeepSeek API密钥
                - deepseek_base_url: DeepSeek API基础URL (可选)
                - semantic_scholar_api_key: Semantic Scholar API密钥 (可选)
                - output_dir: 输出目录 (可选，默认为 'filtered_papers')
                - debug: 是否启用调试模式 (可选，默认为False)
                - max_concurrent_requests: 最大并发请求数 (可选，默认为10)
                - request_delay: 请求间延迟(秒) (可选，默认为0.1)
        """
        self.deepseek_api_key = config.get('deepseek_api_key')
        self.deepseek_base_url = config.get('deepseek_base_url', 'https://api.deepseek.com/v1')
        self.semantic_scholar_api_key = config.get('semantic_scholar_api_key', '')
        self.output_dir = config.get('output_dir', 'filtered_papers')
        self.debug = config.get('debug', False)
        self.max_concurrent_requests = config.get('max_concurrent_requests', 10)
        self.request_delay = config.get('request_delay', 0.1)
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 已处理论文集合，避免重复处理
        self.processed_papers: Set[str] = set()
        
        # 创建信号量来限制并发请求
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API密钥是必需的")
    
    def log_debug(self, message: str):
        """调试日志"""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def extract_arxiv_id(self, url: str) -> str:
        """从arxiv URL中提取文章ID"""
        patterns = [
            r'arxiv\.org/abs/([\d\.]+)',
            r'arxiv\.org/pdf/([\d\.]+)',
            r'alphaxiv\.org/html/([\d\.]+v?\d*)',
            r'([\d]{4}\.[\d]{4,5}v?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                arxiv_id = match.group(1)
                # 移除版本号后缀
                arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
                return arxiv_id
        return ""
    
    async def get_paper_citations(self, arxiv_url: str) -> Optional[Dict[str, Any]]:
        """获取论文的引用关系"""
        self.log_debug(f"获取论文引用关系: {arxiv_url}")
        
        try:
            from arxiv_mcp_server import ArxivMCPServer
            server = ArxivMCPServer(debug_mode=self.debug)
            result = await server.analyze_paper_citations(arxiv_url)
            return result
        except Exception as e:
            self.log_debug(f"获取引用关系时出错: {e}")
            return None
    
    async def check_paper_relevance(self, title: str, abstract: str, keyword: str) -> Dict[str, Any]:
        """
        使用DeepSeek API检查论文与关键字的相关性（带并发限制）
        
        Returns:
            dict: 包含 'relevant' (bool), 'score' (float), 'explanation' (str)
        """
        async with self.semaphore:  # 限制并发请求
            # 添加请求延迟
            await asyncio.sleep(self.request_delay)
            
            self.log_debug(f"检查论文相关性: {title[:50]}...")
            
            prompt = f"""请分析以下论文是否与关键字 "{keyword}" 相关，并给出相关性评分。

论文标题: {title}

论文摘要: {abstract}

请从以下方面分析:
1. 论文是否直接提到或使用了关键字相关的技术
2. 论文的主要贡献是否与关键字相关
3. 论文是否对关键字相关的技术进行了改进或扩展

请输出JSON格式的结果:
{{
    "relevant": true/false,
    "score": 0.0-1.0,
    "explanation": "详细解释为什么相关或不相关，包括具体的证据"
}}

只需要输出JSON，不需要其他内容。使用中文输出。"""

            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.deepseek_base_url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.deepseek_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500
                    }
                    
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            content = result['choices'][0]['message']['content']
                            
                            # 尝试解析JSON
                            try:
                                # 提取JSON部分
                                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    relevance_result = json.loads(json_str)
                                    
                                    # 验证结果格式
                                    if all(key in relevance_result for key in ['relevant', 'score', 'explanation']):
                                        return relevance_result
                                    else:
                                        self.log_debug(f"DeepSeek返回的JSON格式不正确: {relevance_result}")
                                        return {"relevant": False, "score": 0.0, "explanation": "解析结果格式错误"}
                                else:
                                    self.log_debug(f"无法从DeepSeek回复中提取JSON: {content}")
                                    return {"relevant": False, "score": 0.0, "explanation": "无法解析API回复"}
                            except json.JSONDecodeError as e:
                                self.log_debug(f"JSON解析错误: {e}")
                                return {"relevant": False, "score": 0.0, "explanation": "JSON解析失败"}
                        else:
                            self.log_debug(f"DeepSeek API请求失败: {response.status}")
                            return {"relevant": False, "score": 0.0, "explanation": "API请求失败"}
                            
            except Exception as e:
                self.log_debug(f"检查论文相关性时出错: {e}")
                return {"relevant": False, "score": 0.0, "explanation": f"检查过程中出错: {str(e)}"}
    
    async def get_arxiv_paper_info(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """获取arxiv论文的基本信息"""
        self.log_debug(f"获取ArXiv论文信息: {arxiv_id}")
        
        async with aiohttp.ClientSession() as session:
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self.parse_arxiv_response(content)
                    else:
                        self.log_debug(f"ArXiv API请求失败: {response.status}")
                        return None
            except Exception as e:
                self.log_debug(f"获取ArXiv信息时出错: {e}")
                return None
    
    def parse_arxiv_response(self, xml_content: str) -> Optional[Dict[str, Any]]:
        """解析arxiv API响应"""
        try:
            root = ET.fromstring(xml_content)
            entry = root.find('.//{http://www.w3.org/2005/Atom}entry')
            
            if entry is not None:
                title = entry.find('.//{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('.//{http://www.w3.org/2005/Atom}summary').text.strip()
                
                authors = []
                for author in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
                    name = author.find('.//{http://www.w3.org/2005/Atom}name').text
                    authors.append(name)
                
                published = entry.find('.//{http://www.w3.org/2005/Atom}published').text
                arxiv_id = entry.find('.//{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                url = entry.find('.//{http://www.w3.org/2005/Atom}id').text
                
                return {
                    'title': title,
                    'abstract': summary,
                    'authors': authors,
                    'published': published,
                    'arxiv_id': arxiv_id,
                    'url': url
                }
            return None
        except Exception as e:
            self.log_debug(f"解析ArXiv响应时出错: {e}")
            return None
    
    async def filter_papers_by_keyword(self, source_papers: List[str], keyword: str) -> List[FilteredPaper]:
        """
        根据关键字筛选论文（并行处理）
        
        Args:
            source_papers: 源论文URL列表
            keyword: 筛选关键字
            
        Returns:
            List[FilteredPaper]: 筛选后的论文列表
        """
        all_papers_to_process = []
        source_paper_tasks = []
        
        # 首先处理源论文
        for source_paper_url in source_papers:
            self.log_debug(f"处理源论文: {source_paper_url}")
            
            # 提取arxiv ID
            source_arxiv_id = self.extract_arxiv_id(source_paper_url)
            if not source_arxiv_id:
                self.log_debug(f"无法提取arxiv ID: {source_paper_url}")
                continue
            
            # 添加源论文到处理列表
            if source_arxiv_id not in self.processed_papers:
                source_paper_tasks.append(self.process_source_paper(source_paper_url, source_arxiv_id, keyword))
                self.processed_papers.add(source_arxiv_id)
            
            # 获取引用关系
            citation_data = await self.get_paper_citations(source_paper_url)
            if not citation_data or 'papers' not in citation_data:
                self.log_debug(f"无法获取引用数据: {source_paper_url}")
                continue
            
            papers = citation_data['papers']
            
            # 收集引用该论文的文章
            citing_papers = papers.get('citing_papers', [])
            for paper in citing_papers:
                arxiv_id = paper.get('externalIds', {}).get('ArXiv')
                if arxiv_id and arxiv_id not in self.processed_papers:
                    all_papers_to_process.append((paper, arxiv_id, source_arxiv_id, 'citing'))
                    self.processed_papers.add(arxiv_id)
            
            # 收集该论文引用的文章
            referenced_papers = papers.get('referenced_papers', [])
            for paper in referenced_papers:
                arxiv_id = paper.get('externalIds', {}).get('ArXiv')
                if arxiv_id and arxiv_id not in self.processed_papers:
                    all_papers_to_process.append((paper, arxiv_id, source_arxiv_id, 'referenced'))
                    self.processed_papers.add(arxiv_id)
        
        print(f"找到 {len(source_paper_tasks)} 篇源论文和 {len(all_papers_to_process)} 篇关联论文需要处理，开始并行处理...")
        
        # 创建所有处理任务
        tasks = source_paper_tasks.copy()
        for paper_data, arxiv_id, source_arxiv_id, relationship in all_papers_to_process:
            task = self.process_paper(paper_data, arxiv_id, keyword, source_arxiv_id, relationship)
            tasks.append(task)
        
        # 并行执行所有任务
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 过滤结果
        filtered_papers = []
        for result in results:
            if isinstance(result, FilteredPaper):
                filtered_papers.append(result)
            elif isinstance(result, Exception):
                self.log_debug(f"处理论文时出错: {result}")
        
        print(f"并行处理完成，耗时: {end_time - start_time:.2f}秒")
        print(f"找到 {len(filtered_papers)} 篇相关论文（包含源论文）")
        
        return filtered_papers
    
    async def process_source_paper(self, source_paper_url: str, arxiv_id: str, keyword: str) -> Optional[FilteredPaper]:
        """
        处理源论文（原论文）
        
        Args:
            source_paper_url: 源论文URL
            arxiv_id: ArXiv ID
            keyword: 关键字
            
        Returns:
            Optional[FilteredPaper]: 处理后的源论文信息
        """
        try:
            self.log_debug(f"处理源论文: {arxiv_id}")
            
            # 获取ArXiv论文信息
            arxiv_info = await self.get_arxiv_paper_info(arxiv_id)
            if not arxiv_info:
                self.log_debug(f"无法获取源论文信息: {arxiv_id}")
                return None
            
            title = arxiv_info.get('title', '')
            abstract = arxiv_info.get('abstract', '')
            
            if not title or not abstract:
                self.log_debug(f"跳过源论文（缺少标题或摘要）: {arxiv_id}")
                return None
            
            # 检查相关性
            relevance_result = await self.check_paper_relevance(title, abstract, keyword)
            
            # 源论文总是包含在结果中，但会标记其相关性
            self.log_debug(f"源论文相关性: {title[:50]}... (评分: {relevance_result['score']})")
            
            return FilteredPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=arxiv_info.get('authors', []),
                published=arxiv_info.get('published', ''),
                url=arxiv_info.get('url', source_paper_url),
                relevance_score=relevance_result['score'],
                relevance_explanation=relevance_result['explanation'],
                source_paper=arxiv_id,  # 源论文的source_paper指向自己
                relationship='source'   # 标记为源论文
            )
            
        except Exception as e:
            self.log_debug(f"处理源论文 {arxiv_id} 时出错: {e}")
            return None
    
    async def process_paper(self, paper_data: Dict[str, Any], arxiv_id: str, keyword: str, 
                          source_arxiv_id: str, relationship: str) -> Optional[FilteredPaper]:
        """处理单个论文"""
        try:
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            
            # 如果没有摘要，尝试从ArXiv API获取
            if not abstract:
                arxiv_info = await self.get_arxiv_paper_info(arxiv_id)
                if arxiv_info:
                    abstract = arxiv_info.get('abstract', '')
                    if not title:
                        title = arxiv_info.get('title', '')
            
            if not title or not abstract:
                self.log_debug(f"跳过论文（缺少标题或摘要）: {arxiv_id}")
                return None
            
            # 检查相关性
            relevance_result = await self.check_paper_relevance(title, abstract, keyword)
            
            if relevance_result['relevant']:
                self.log_debug(f"发现相关论文: {title[:50]}... (评分: {relevance_result['score']})")
                
                # 获取完整的ArXiv信息
                arxiv_info = await self.get_arxiv_paper_info(arxiv_id)
                if not arxiv_info:
                    authors = paper_data.get('authors', [])
                    if isinstance(authors, list) and len(authors) > 0 and isinstance(authors[0], dict):
                        authors = [author.get('name', '') for author in authors]
                    published = paper_data.get('year', '')
                    url = f"https://arxiv.org/abs/{arxiv_id}"
                else:
                    authors = arxiv_info.get('authors', [])
                    published = arxiv_info.get('published', '')
                    url = arxiv_info.get('url', f"https://arxiv.org/abs/{arxiv_id}")
                
                return FilteredPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published=published,
                    url=url,
                    relevance_score=relevance_result['score'],
                    relevance_explanation=relevance_result['explanation'],
                    source_paper=source_arxiv_id,
                    relationship=relationship
                )
            
            return None
            
        except Exception as e:
            self.log_debug(f"处理论文 {arxiv_id} 时出错: {e}")
            return None
    
    def save_filtered_papers(self, filtered_papers: List[FilteredPaper], keyword: str):
        """保存筛选后的论文（简化版，只保存URL和简介）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"filtered_papers_{keyword.replace(' ', '_')}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 转换为简化的可序列化格式
        papers_data = []
        for paper in filtered_papers:
            papers_data.append({
                'arxiv_id': paper.arxiv_id,
                'title': paper.title,
                'abstract': paper.abstract,  # 简介
                'authors': paper.authors,
                'published': paper.published,
                'url': paper.url,  # 只保存URL，不下载论文
                'relevance_score': paper.relevance_score,
                'relevance_explanation': paper.relevance_explanation,
                'source_paper': paper.source_paper,
                'relationship': paper.relationship
            })
        
        save_data = {
            'keyword': keyword,
            'timestamp': datetime.now().isoformat(),
            'total_papers': len(filtered_papers),
            'processing_settings': {
                'max_concurrent_requests': self.max_concurrent_requests,
                'request_delay': self.request_delay
            },
            'papers': papers_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"筛选结果已保存到: {filepath}")
        print(f"找到 {len(filtered_papers)} 篇相关论文")
        
        # 生成简化的摘要文件
        self.save_summary_file(filtered_papers, keyword, timestamp)
    
    def save_summary_file(self, filtered_papers: List[FilteredPaper], keyword: str, timestamp: str):
        """保存为 Markdown 格式的简洁摘要文件，并过滤特殊字符和避免触发代码块"""
        summary_filename = f"paper_summary_{keyword.replace(' ', '_')}_{timestamp}.md"
        summary_filepath = os.path.join(self.output_dir, summary_filename)

        # Markdown 转义函数（不过度使用反斜杠，避免造成渲染异常）
        def escape_md(text: str) -> str:
            return re.sub(r'([`*_{}\[\]()#+!])', r'\\\1', text)

        source_papers = [p for p in filtered_papers if p.relationship == 'source']
        citing_papers = [p for p in filtered_papers if p.relationship == 'citing']
        referenced_papers = [p for p in filtered_papers if p.relationship == 'referenced']
        sorted_papers = sorted(filtered_papers, key=lambda x: x.relevance_score, reverse=True)

        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 论文筛选摘要\n\n")
            f.write(f"**关键词**: `{escape_md(keyword)}`  \n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**总论文数**: {len(filtered_papers)}\n\n")

            f.write(f"- 源论文: **{len(source_papers)} 篇**\n")
            f.write(f"- 引用源论文的文章: **{len(citing_papers)} 篇**\n")
            f.write(f"- 源论文引用的文章: **{len(referenced_papers)} 篇**\n\n")

            high = [p for p in sorted_papers if p.relevance_score >= 0.7]
            medium = [p for p in sorted_papers if 0.5 <= p.relevance_score < 0.7]
            low = [p for p in sorted_papers if p.relevance_score < 0.5]

            f.write("## 按相关性评分统计\n\n")
            f.write(f"- 高相关性 (≥0.7): **{len(high)} 篇**\n")
            f.write(f"- 中相关性 (0.5–0.7): **{len(medium)} 篇**\n")
            f.write(f"- 低相关性 (<0.5): **{len(low)} 篇**\n\n")

            if source_papers:
                f.write("## 源论文列表\n\n")
                for i, paper in enumerate(sorted(source_papers, key=lambda x: x.relevance_score, reverse=True), 1):
                    title = escape_md(paper.title)
                    arxiv_id = escape_md(paper.arxiv_id)
                    url = paper.url
                    authors = escape_md(', '.join(paper.authors[:3])) + (' 等' if len(paper.authors) > 3 else '')
                    abstract = escape_md(paper.abstract[:200] + '...')
                    explanation = escape_md(paper.relevance_explanation)

                    f.write(f"#### {i}. {title}\n")
                    f.write(f"- **ArXiv ID**: `{arxiv_id}`  \n")
                    f.write(f"- **URL**: [{url}]({url})  \n")
                    f.write(f"- **评分**: `{paper.relevance_score:.2f}`  \n")
                    f.write(f"- **作者**: {authors}  \n")
                    f.write(f"- **摘要**: {abstract}  \n")
                    f.write(f"- **相关性说明**: {explanation}  \n\n")

            f.write("## 所有论文（按相关性排序）\n\n")
            for i, paper in enumerate(sorted_papers, 1):
                title = escape_md(paper.title)
                relationship = escape_md({
                    'source': '源论文',
                    'citing': '引用源论文',
                    'referenced': '被源论文引用'
                }.get(paper.relationship, paper.relationship))
                arxiv_id = escape_md(paper.arxiv_id)
                authors = escape_md(', '.join(paper.authors[:3])) + (' 等' if len(paper.authors) > 3 else '')
                abstract = escape_md(paper.abstract[:200] + '...')
                explanation = escape_md(paper.relevance_explanation)

                f.write(f"#### {i}. [{title}]({paper.url})\n")
                f.write(f"- **类型**: `{relationship}`  \n")
                f.write(f"- **ArXiv ID**: `{arxiv_id}`  \n")
                f.write(f"- **评分**: `{paper.relevance_score:.2f}`  \n")
                if paper.relationship != 'source':
                    source_paper = escape_md(paper.source_paper)
                    f.write(f"- **关联源论文**: `{source_paper}`  \n")
                f.write(f"- **作者**: {authors}  \n")
                f.write(f"- **摘要**: {abstract}  \n")
                f.write(f"- **相关性说明**: {explanation}  \n\n")

        print(f"Markdown 摘要文件已保存: {summary_filepath}")
    
    def print_summary(self, filtered_papers: List[FilteredPaper], keyword: str):
        """打印筛选结果摘要"""
        print(f"\n=== 筛选结果摘要 ===")
        print(f"关键字: {keyword}")
        print(f"总共找到: {len(filtered_papers)} 篇论文")
        
        # 按类型分类
        source_papers = [p for p in filtered_papers if p.relationship == 'source']
        citing_papers = [p for p in filtered_papers if p.relationship == 'citing']
        referenced_papers = [p for p in filtered_papers if p.relationship == 'referenced']
        
        print(f"- 源论文: {len(source_papers)} 篇")
        print(f"- 引用源论文的文章: {len(citing_papers)} 篇")
        print(f"- 源论文引用的文章: {len(referenced_papers)} 篇")
        
        # 按评分排序
        sorted_papers = sorted(filtered_papers, key=lambda x: x.relevance_score, reverse=True)
        
        # 统计信息
        high_relevance = [p for p in sorted_papers if p.relevance_score >= 0.7]
        medium_relevance = [p for p in sorted_papers if 0.5 <= p.relevance_score < 0.7]
        low_relevance = [p for p in sorted_papers if p.relevance_score < 0.5]
        
        print(f"\n按相关性分类:")
        print(f"- 高相关性 (≥0.7): {len(high_relevance)} 篇")
        print(f"- 中等相关性 (0.5-0.7): {len(medium_relevance)} 篇")
        print(f"- 低相关性 (<0.5): {len(low_relevance)} 篇")
        
        print(f"\n=== 前10篇高相关性论文 ===")
        for i, paper in enumerate(sorted_papers[:10], 1):
            relationship_label = {
                'source': '【源论文】',
                'citing': '【引用源论文】',
                'referenced': '【被源论文引用】'
            }.get(paper.relationship, f'【{paper.relationship}】')
            
            print(f"{i}. {relationship_label} {paper.title}")
            print(f"   ArXiv ID: {paper.arxiv_id}")
            print(f"   URL: {paper.url}")
            print(f"   评分: {paper.relevance_score:.2f}")
            if paper.relationship != 'source':
                print(f"   关系: {paper.relationship} (来源: {paper.source_paper})")
            print(f"   相关性说明: {paper.relevance_explanation[:100]}...")
            print()

async def main():
    # 配置
    config = {
        'deepseek_api_key': 'sk-7af208a449c549fdaa60dc3b6232f4d0',  # API密钥
        'semantic_scholar_api_key': 'NULL',  # 可选
        'output_dir': 'filtered_papers',
        'debug': True,
        'max_concurrent_requests': 15,  # 增加并发请求数
        'request_delay': 0.05  # 减少请求延迟
    }
    
    # 源论文列表
    source_papers = [
        "https://arxiv.org/abs/2309.06180",  # vLLM
    ]
    
    # 关键字
    keyword = 'KV-cache'
    
    # 创建筛选器
    filter_system = ArxivCitationFilter(config)
    
    # 执行筛选
    print("开始筛选论文...")
    start_time = time.time()
    filtered_papers = await filter_system.filter_papers_by_keyword(source_papers, keyword)
    end_time = time.time()
    
    print(f"总处理时间: {end_time - start_time:.2f}秒")
    
    # 保存结果
    filter_system.save_filtered_papers(filtered_papers, keyword)
    
    # 打印摘要
    filter_system.print_summary(filtered_papers, keyword)

if __name__ == "__main__":
    asyncio.run(main())