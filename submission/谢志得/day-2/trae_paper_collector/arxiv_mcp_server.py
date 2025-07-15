import json
import os
from datetime import datetime
from typing import List, Dict, Any
from fastmcp import FastMCP
import aiohttp
import asyncio
import re
from dataclasses import dataclass, field

@dataclass
class ArxivPaper:
    title: str
    abstract: str
    url: str
    authors: List[str]
    published: str
    arxiv_id: str

    def to_dict(self):
        return {
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "authors": self.authors,
            "published": self.published,
            "arxiv_id": self.arxiv_id
        }

@dataclass
class SemanticScholarPaper:
    paperId: str
    title: str
    abstract: str = ""
    externalIds: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "paperId": self.paperId,
            "title": self.title,
            "abstract": self.abstract,
            "externalIds": self.externalIds,
            "ArXiv": self.externalIds.get("ArXiv")
        }


# Create FastMCP server instance
mcp = FastMCP("arxiv-citation-analyzer")

class ArxivMCPServer:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode  # 接收debug参数
        # 移除config文件依赖，使用环境变量或默认值
        self.semantic_scholar_api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY', '')
        self.semantic_scholar_api_url = "https://api.semanticscholar.org/graph/v1/paper/"
    
    async def _handle_rate_limit_retry(self, session, method, url, max_retries=3, **kwargs):
        """处理API请求的重试逻辑，特别是429错误"""
        for attempt in range(max_retries + 1):
            try:
                if method.lower() == 'get':
                    response = await session.get(url, **kwargs)
                elif method.lower() == 'post':
                    response = await session.post(url, **kwargs)
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")
                
                if response.status == 429:
                    if attempt < max_retries:
                        # 指数退避策略：2^attempt * 2 秒
                        wait_time = (2 ** attempt) * 2
                        if self.debug_mode:
                            print(f"[DEBUG] 遇到429错误，等待 {wait_time} 秒后重试 (尝试 {attempt + 1}/{max_retries + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        if self.debug_mode:
                            print(f"[DEBUG] 达到最大重试次数，429错误处理失败")
                        return response
                else:
                    return response
                    
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 2
                    if self.debug_mode:
                        print(f"[DEBUG] 请求异常: {e}，等待 {wait_time} 秒后重试")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise e
        
        return None
    
    async def analyze_paper_citations(self, arxiv_url: str) -> Dict[str, Any]:
        """分析arxiv文章的引用关系"""
        import json
        import os
        from datetime import datetime
        import glob
        
        if self.debug_mode:
            print(f"[DEBUG] 开始分析论文引用关系")
            print(f"[DEBUG] 输入URL: {arxiv_url}")
        
        # 提取arxiv ID
        arxiv_id = self.extract_arxiv_id(arxiv_url)
        if self.debug_mode:
            print(f"[DEBUG] 提取的ArXiv ID: {arxiv_id}")
        if not arxiv_id:
            if self.debug_mode:
                print(f"[DEBUG] 错误: 无效的arxiv链接")
            return {"error": "无效的arxiv链接"}
        
        # 检查缓存：查找json_files目录中是否已有该arxiv_id的文件
        mcp_dir = os.path.dirname(os.path.abspath(__file__))
        json_files_dir = os.path.join(mcp_dir, "json_files")
        
        if os.path.exists(json_files_dir):
            # 查找匹配的文件（格式：arxiv_{id}_{timestamp}.json）
            pattern = f"arxiv_{arxiv_id.replace('.', '_')}_*.json"
            cache_files = glob.glob(os.path.join(json_files_dir, pattern))
            
            if cache_files:
                # 找到缓存文件，选择最新的一个
                latest_cache = max(cache_files, key=os.path.getmtime)
                if self.debug_mode:
                    print(f"[DEBUG] 找到缓存文件: {latest_cache}")
                
                try:
                    with open(latest_cache, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    # 检查缓存数据是否有效（包含papers字段且不是错误）
                    if 'papers' in cached_data and 'error' not in cached_data:
                        if self.debug_mode:
                            print(f"[DEBUG] 使用缓存数据，跳过API请求")
                        return {
                            "papers": cached_data["papers"],
                            "summary": cached_data.get("summary", {})
                        }
                    else:
                        if self.debug_mode:
                            print(f"[DEBUG] 缓存数据无效，继续执行API请求")
                except Exception as e:
                    if self.debug_mode:
                        print(f"[DEBUG] 读取缓存文件失败: {e}，继续执行API请求")
            else:
                if self.debug_mode:
                    print(f"[DEBUG] 未找到缓存文件，执行API请求")
        else:
            if self.debug_mode:
                print(f"[DEBUG] json_files目录不存在，执行API请求")
        
        # 获取主文章信息
        if self.debug_mode:
            print(f"[DEBUG] 开始获取主文章信息...")
        main_paper = await self.get_arxiv_paper(arxiv_id)
        if self.debug_mode:
            print(f"[DEBUG] 主文章获取结果: {type(main_paper)} - {main_paper}")
        if not main_paper:
            if self.debug_mode:
                print(f"[DEBUG] 错误: 无法获取文章信息")
            return {"error": "无法获取文章信息"}
        
        # 获取引用和被引用文章的基本信息（不包含摘要）
        if self.debug_mode:
            print(f"[DEBUG] 开始获取引用该论文的文章...")
        citing_papers_basic = await self.get_citing_papers_basic(arxiv_id)
        if self.debug_mode:
            print(f"[DEBUG] 引用文章基本信息获取结果: {type(citing_papers_basic)} - 数量: {len(citing_papers_basic) if citing_papers_basic else 0}")
        
        if self.debug_mode:
            print(f"[DEBUG] 开始获取该论文引用的文章...")
        referenced_papers_basic = await self.get_referenced_papers_basic(arxiv_id)
        if self.debug_mode:
            print(f"[DEBUG] 被引用文章基本信息获取结果: {type(referenced_papers_basic)} - 数量: {len(referenced_papers_basic) if referenced_papers_basic else 0}")
        
        # 合并所有论文ID，进行一次批量详情获取
        all_paper_ids = []
        if citing_papers_basic:
            all_paper_ids.extend([paper.paperId for paper in citing_papers_basic if paper.paperId])
        if referenced_papers_basic:
            all_paper_ids.extend([paper.paperId for paper in referenced_papers_basic if paper.paperId])
        
        if self.debug_mode:
            print(f"[DEBUG] 合并后需要获取详情的论文总数: {len(all_paper_ids)}")
        
        # 一次性批量获取所有论文的详细信息
        papers_details = {}
        if all_paper_ids:
            if self.debug_mode:
                print(f"[DEBUG] 开始批量获取所有论文的详细信息...")
            papers_with_details = await self.get_papers_batch_details(all_paper_ids)
            
            # 建立ID到详情的映射
            for i, paper_id in enumerate(all_paper_ids):
                if i < len(papers_with_details) and papers_with_details[i]:
                    papers_details[paper_id] = papers_with_details[i]
        
        # 更新引用论文的详细信息
        citing_papers = []
        if citing_papers_basic:
            for paper in citing_papers_basic:
                if paper.paperId in papers_details:
                    detail = papers_details[paper.paperId]
                    paper.abstract = detail.abstract
                    paper.externalIds = detail.externalIds
                citing_papers.append(paper)
        
        # 更新被引用论文的详细信息
        referenced_papers = []
        if referenced_papers_basic:
            for paper in referenced_papers_basic:
                if paper.paperId in papers_details:
                    detail = papers_details[paper.paperId]
                    paper.abstract = detail.abstract
                    paper.externalIds = detail.externalIds
                referenced_papers.append(paper)
        
        if self.debug_mode:
            print(f"[DEBUG] 最终引用文章数量: {len(citing_papers)}")
            print(f"[DEBUG] 最终被引用文章数量: {len(referenced_papers)}")
        
        # 汇总所有文章
        if self.debug_mode:
            print(f"[DEBUG] 开始汇总所有文章信息...")
        try:
            main_paper_dict = main_paper.to_dict() if main_paper else None
            if self.debug_mode:
                print(f"[DEBUG] 主文章字典转换结果: {type(main_paper_dict)}")
            
            citing_papers_dict = [paper.to_dict() for paper in citing_papers] if citing_papers else []
            if self.debug_mode:
                print(f"[DEBUG] 引用文章字典转换结果: {len(citing_papers_dict)} 篇")
            
            referenced_papers_dict = [paper.to_dict() for paper in referenced_papers] if referenced_papers else []
            if self.debug_mode:
                print(f"[DEBUG] 被引用文章字典转换结果: {len(referenced_papers_dict)} 篇")
            
            all_papers = {
                "main_paper": main_paper_dict,
                "citing_papers": citing_papers_dict,
                "referenced_papers": referenced_papers_dict
            }
            if self.debug_mode:
                print(f"[DEBUG] 所有文章汇总完成: {type(all_papers)}")
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] 文章汇总时出错: {e}")
                print(f"[DEBUG] main_paper类型: {type(main_paper)}")
                print(f"[DEBUG] citing_papers类型: {type(citing_papers)}")
                print(f"[DEBUG] referenced_papers类型: {type(referenced_papers)}")
            return {"error": f"文章汇总时出错: {e}"}
        
        # 保存到本地JSON文件（仅在成功获取数据时保存）
        if self.debug_mode:
            print(f"[DEBUG] 开始保存到本地JSON文件...")
        
        # 检查是否成功获取了数据
        has_valid_data = (
            main_paper and 
            (citing_papers is not None or referenced_papers is not None) and
            all_papers and 'main_paper' in all_papers
        )
        
        if has_valid_data:
            try:
                # 创建保存数据的结构
                save_data = {
                    "arxiv_id": arxiv_id,
                    "arxiv_url": arxiv_url,
                    "timestamp": datetime.now().isoformat(),
                    "papers": all_papers,
                    "summary": {
                        "total_citing": len(citing_papers) if citing_papers else 0,
                        "total_referenced": len(referenced_papers) if referenced_papers else 0
                    }
                }
                if self.debug_mode:
                    print(f"[DEBUG] 保存数据结构创建完成: {type(save_data)}")
                
                # 确保MCP目录存在
                mcp_dir = os.path.dirname(os.path.abspath(__file__))
                json_files_dir = os.path.join(mcp_dir, "json_files")
                if self.debug_mode:
                    print(f"[DEBUG] MCP目录: {mcp_dir}")
                    print(f"[DEBUG] JSON文件目录: {json_files_dir}")
                
                # 如果json_files目录不存在，则创建
                if not os.path.exists(json_files_dir):
                    os.makedirs(json_files_dir)
                    if self.debug_mode:
                        print(f"[DEBUG] 已创建目录: {json_files_dir}")

                # 生成文件名（使用arxiv_id和时间戳）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"arxiv_{arxiv_id.replace('.', '_')}_{timestamp}.json"
                filepath = os.path.join(json_files_dir, filename)
                if self.debug_mode:
                    print(f"[DEBUG] 生成的文件路径: {filepath}")
                
                # 保存到文件
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                
                if self.debug_mode:
                    print(f"[INFO] 相关论文信息已保存到: {filepath}")
                    print(f"[DEBUG] 文件保存成功")
                
            except Exception as e:
                if self.debug_mode:
                    print(f"[ERROR] 保存文件时出错: {str(e)}")
                    print(f"[DEBUG] 保存文件异常详情: {type(e).__name__}: {e}")
        else:
            if self.debug_mode:
                print(f"[DEBUG] 数据获取不完整或失败，跳过文件保存")
        
        # 返回检索结果，不进行智能分析
        if self.debug_mode:
            print(f"[DEBUG] 开始构建返回结果...")
        try:
            result = {
                "papers": all_papers,
                "summary": {
                    "total_citing": len(citing_papers) if citing_papers else 0,
                    "total_referenced": len(referenced_papers) if referenced_papers else 0
                }
            }
            if self.debug_mode:
                print(f"[DEBUG] 返回结果构建完成: {type(result)}")
                print(f"[DEBUG] 返回结果keys: {list(result.keys())}")
                if 'papers' in result:
                    print(f"[DEBUG] papers keys: {list(result['papers'].keys()) if result['papers'] else 'None'}")
            return result
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] 构建返回结果时出错: {e}")
            return {"error": f"构建返回结果时出错: {e}"}
    
    def extract_arxiv_id(self, url: str) -> str:
        """从arxiv URL或alphaxiv URL中提取文章ID"""
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
                # 移除版本号后缀（如v1, v2等）
                arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
                return arxiv_id
        return ""
    
    async def get_paper_details_from_semantic_scholar(self, paper_id: str) -> SemanticScholarPaper | None:
        """从Semantic Scholar获取论文详细信息，包括摘要和外部ID"""
        if self.debug_mode:
            print(f"[DEBUG] 从Semantic Scholar获取论文详情，paperId: {paper_id}")
        
        # 使用默认的Semantic Scholar API URL
        api_url = self.semantic_scholar_api_url
        
        async with aiohttp.ClientSession() as session:
            url = f"{api_url}{paper_id}"
            params = {
                "fields": "abstract,externalIds,title"
            }
            
            # 检查是否有API密钥配置
            headers = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
                if self.debug_mode:
                    print(f"[DEBUG] 使用API密钥获取论文详情")
            elif self.debug_mode:
                print(f"[DEBUG] 使用公开访问获取论文详情")
            
            if self.debug_mode:
                print(f"[DEBUG] 请求URL: {url}")
                print(f"[DEBUG] 请求参数: {params}")
            
            try:
                response = await self._handle_rate_limit_retry(
                    session, 'get', url, params=params, headers=headers
                )
                
                if response is None:
                    if self.debug_mode:
                        print(f"[DEBUG] 请求失败，无法获取响应")
                    return None
                
                async with response:
                    if self.debug_mode:
                        print(f"[DEBUG] 响应状态码: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        if self.debug_mode:
                            print(f"[DEBUG] 成功获取论文详情: {data.get('title', 'N/A')[:50]}...")
                        
                        if data.get("paperId"):
                            return SemanticScholarPaper(
                                paperId=data.get("paperId"),
                                title=data.get("title"),
                                abstract=data.get("abstract", ""),
                                externalIds=data.get("externalIds", {})
                            )
                        else:
                            if self.debug_mode:
                                print(f"[DEBUG] 未找到 paperId 为 {paper_id} 的论文详情。")
                            return None
                    elif response.status == 429:
                        if self.debug_mode:
                            print(f"[DEBUG] API请求频率限制，已达到最大重试次数")
                        return None
                    else:
                        if self.debug_mode:
                            print(f"[DEBUG] API请求失败，状态码: {response.status}")
                        return None
                        
            except aiohttp.ClientError as e:
                if self.debug_mode:
                    print(f"[DEBUG] 从Semantic Scholar获取论文详情时发生HTTP错误: {e}")
                return None
            except Exception as e:
                if self.debug_mode:
                    print(f"[DEBUG] 从Semantic Scholar获取论文详情时发生未知错误: {e}")
                return None
    
    async def get_papers_batch_details(self, paper_ids: List[str]) -> List[SemanticScholarPaper | None]:
        """批量获取多个论文的详细信息，包括摘要和外部ID"""
        if self.debug_mode:
            print(f"[DEBUG] 批量获取 {len(paper_ids)} 篇论文的详细信息")
        
        if not paper_ids:
            return []
        
        # Semantic Scholar批量API每次最多支持500个ID，这里分批处理
        batch_size = 500
        all_results = []
        
        for i in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[i:i + batch_size]
            if self.debug_mode:
                print(f"[DEBUG] 处理批次 {i//batch_size + 1}，包含 {len(batch_ids)} 个ID")
            
            batch_results = await self._get_batch_papers_details(batch_ids)
            all_results.extend(batch_results)
        
        return all_results
    
    async def _get_batch_papers_details(self, paper_ids: List[str]) -> List[SemanticScholarPaper | None]:
        """获取单个批次的论文详细信息"""
        async with aiohttp.ClientSession() as session:
            url = "https://api.semanticscholar.org/graph/v1/paper/batch"
            
            # 检查是否有API密钥配置
            headers = {
                "Content-Type": "application/json"
            }
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
                if self.debug_mode:
                    print(f"[DEBUG] 使用API密钥进行批量请求")
            elif self.debug_mode:
                print(f"[DEBUG] 使用公开访问进行批量请求")
            
            # 构建请求体
            payload = {
                "ids": paper_ids
            }
            
            params = {
                "fields": "paperId,title,abstract,externalIds"
            }
            
            if self.debug_mode:
                print(f"[DEBUG] 批量请求URL: {url}")
                print(f"[DEBUG] 批量请求参数: {params}")
                print(f"[DEBUG] 批量请求体包含 {len(paper_ids)} 个ID")
            
            try:
                response = await self._handle_rate_limit_retry(
                    session, 'post', url, json=payload, params=params, headers=headers
                )
                
                if response is None:
                    if self.debug_mode:
                        print(f"[DEBUG] 批量请求失败，无法获取响应")
                    return [None] * len(paper_ids)
                
                async with response:
                    if self.debug_mode:
                        print(f"[DEBUG] 批量请求响应状态码: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        if self.debug_mode:
                            print(f"[DEBUG] 批量请求成功，返回 {len(data)} 个结果")
                        
                        # 保存原始回复到本地文件
                        await self._save_batch_response_to_file(data, paper_ids)
                        
                        results = []
                        for i, paper_data in enumerate(data):
                            if paper_data and paper_data.get("paperId"):
                                paper = SemanticScholarPaper(
                                    paperId=paper_data.get("paperId"),
                                    title=paper_data.get("title", ""),
                                    abstract=paper_data.get("abstract", ""),
                                    externalIds=paper_data.get("externalIds", {})
                                )
                                results.append(paper)
                                if self.debug_mode:
                                    print(f"[DEBUG] 批量结果 {i+1}: {paper.title[:50] if paper.title else 'N/A'}...")
                            else:
                                results.append(None)
                                if self.debug_mode:
                                    print(f"[DEBUG] 批量结果 {i+1}: 无效数据")
                        
                        return results
                    elif response.status == 429:
                        if self.debug_mode:
                            print(f"[DEBUG] 批量请求频率限制，已达到最大重试次数")
                        return [None] * len(paper_ids)
                    else:
                        response_text = await response.text()
                        if self.debug_mode:
                            print(f"[DEBUG] 批量请求失败，状态码: {response.status}")
                            print(f"[DEBUG] 批量请求错误响应: {response_text[:200]}...")
                        return [None] * len(paper_ids)
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"[DEBUG] 批量请求异常: {str(e)}")
                return [None] * len(paper_ids)
    
    async def _save_batch_response_to_file(self, response_data: List[Dict], paper_ids: List[str]):
        """保存批量请求的原始回复到本地文件"""
        try:
            # 确保MCP目录存在
            mcp_dir = os.path.dirname(os.path.abspath(__file__))
            json_files_dir = os.path.join(mcp_dir, "json_files")
            
            # 如果json_files目录不存在，则创建
            if not os.path.exists(json_files_dir):
                os.makedirs(json_files_dir)
            
            # 生成文件名（使用时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_response_{timestamp}.json"
            filepath = os.path.join(json_files_dir, filename)
            
            # 构建保存数据
            save_data = {
                "timestamp": datetime.now().isoformat(),
                "request_paper_ids": paper_ids,
                "response_count": len(response_data),
                "raw_response": response_data
            }
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            if self.debug_mode:
                print(f"[DEBUG] 批量请求原始回复已保存到: {filepath}")
                
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] 保存批量请求原始回复时出错: {str(e)}")

    async def get_arxiv_paper(self, arxiv_id: str) -> ArxivPaper | None:
        """获取arxiv文章信息"""
        if self.debug_mode:
            print(f"[DEBUG] 获取ArXiv论文信息，ID: {arxiv_id}")
        
        async with aiohttp.ClientSession() as session:
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            if self.debug_mode:
                print(f"[DEBUG] ArXiv API URL: {url}")
            
            try:
                async with session.get(url) as response:
                    if self.debug_mode:
                        print(f"[DEBUG] ArXiv API响应状态码: {response.status}")
                    
                    if response.status == 200:
                        content = await response.text()
                        if self.debug_mode:
                            print(f"[DEBUG] ArXiv API响应长度: {len(content)} 字符")
                        
                        paper = self.parse_arxiv_response(content)
                        if self.debug_mode:
                            if paper:
                                print(f"[DEBUG] 成功解析ArXiv论文: {paper.title[:50]}...")
                            else:
                                print(f"[DEBUG] 解析ArXiv响应失败")
                        return paper
                    else:
                        response_text = await response.text()
                        if self.debug_mode:
                            print(f"[DEBUG] ArXiv API请求失败，状态码: {response.status}")
                            print(f"[DEBUG] ArXiv错误响应: {response_text[:200]}...")
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"[DEBUG] ArXiv API请求异常: {str(e)}")
                
        if self.debug_mode:
            print(f"[DEBUG] 返回None")
        return None
    
    def parse_arxiv_response(self, xml_content: str) -> ArxivPaper:
        """解析arxiv API响应"""
        # 这里需要实现XML解析逻辑
        # 简化示例，实际需要使用xml.etree.ElementTree
        import xml.etree.ElementTree as ET
        
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
            
            return ArxivPaper(
                title=title,
                abstract=summary,
                url=url,
                authors=authors,
                published=published,
                arxiv_id=arxiv_id
            )
        return None
    
    async def get_citing_papers_basic(self, arxiv_id: str) -> List[SemanticScholarPaper]:
        """获取引用该文章的论文基本信息（不包含摘要）"""
        if self.debug_mode:
            print(f"[DEBUG] 开始获取引用论文基本信息，ArXiv ID: {arxiv_id}")
        
        async with aiohttp.ClientSession() as session:
            url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}/citations"
            if self.debug_mode:
                print(f"[DEBUG] 请求URL: {url}")
            
            # 检查是否有API密钥配置
            headers = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
                if self.debug_mode:
                    print(f"[DEBUG] 使用API密钥: {self.semantic_scholar_api_key[:10]}...")
            elif self.debug_mode:
                print(f"[DEBUG] 未配置API密钥，使用公开访问")
            
            try:
                async with session.get(url, headers=headers) as response:
                    if self.debug_mode:
                        print(f"[DEBUG] 响应状态码: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        if self.debug_mode:
                            print(f"[DEBUG] 原始数据长度: {len(data.get('data', []))}")
                        
                        # 收集基本信息
                        citing_papers = []
                        for i, citation in enumerate(data.get('data', [])):
                            if self.debug_mode:
                                print(f"[DEBUG] 处理第 {i+1} 个引用")
                            paper_data = citation.get('citingPaper', {})
                            paper_id = paper_data.get('paperId')
                            if paper_id:
                                paper = SemanticScholarPaper(
                                    paperId=paper_id,
                                    title=paper_data.get('title', ''),
                                    abstract='',  # 暂时为空，后续批量获取
                                    externalIds=paper_data.get('externalIds', {})
                                )
                                citing_papers.append(paper)
                                if self.debug_mode:
                                    print(f"[DEBUG] 引用论文信息: title={paper.title[:50] if paper.title else 'N/A'}...")
                        
                        if self.debug_mode:
                            print(f"[DEBUG] 找到 {len(citing_papers)} 篇引用论文基本信息")
                        return citing_papers
                    else:
                        response_text = await response.text()
                        if self.debug_mode:
                            print(f"[DEBUG] API请求失败，状态码: {response.status}")
                            print(f"[DEBUG] 错误响应: {response_text[:200]}...")
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"[DEBUG] 请求异常: {str(e)}")
                
        if self.debug_mode:
            print(f"[DEBUG] 返回空列表")
        return []

    async def get_referenced_papers_basic(self, arxiv_id: str) -> List[SemanticScholarPaper]:
        """获取该文章引用的论文基本信息（不包含摘要）"""
        if self.debug_mode:
            print(f"[DEBUG] 开始获取被引用论文基本信息，ArXiv ID: {arxiv_id}")
        
        async with aiohttp.ClientSession() as session:
            url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}/references"
            if self.debug_mode:
                print(f"[DEBUG] 请求URL: {url}")
            
            # 检查是否有API密钥配置
            headers = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
                if self.debug_mode:
                    print(f"[DEBUG] 使用API密钥: {self.semantic_scholar_api_key[:10]}...")
            elif self.debug_mode:
                print(f"[DEBUG] 未配置API密钥，使用公开访问")
            
            try:
                async with session.get(url, headers=headers) as response:
                    if self.debug_mode:
                        print(f"[DEBUG] 响应状态码: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        if self.debug_mode:
                            print(f"[DEBUG] 原始数据长度: {len(data.get('data', []))}")
                        
                        # 收集基本信息
                        referenced_papers = []
                        for i, reference in enumerate(data.get('data', [])):
                            if self.debug_mode:
                                print(f"[DEBUG] 处理第 {i+1} 个参考文献")
                            paper_data = reference.get('citedPaper', {})
                            paper_id = paper_data.get('paperId')
                            if paper_id:
                                paper = SemanticScholarPaper(
                                    paperId=paper_id,
                                    title=paper_data.get('title', ''),
                                    abstract='',  # 暂时为空，后续批量获取
                                    externalIds=paper_data.get('externalIds', {})
                                )
                                referenced_papers.append(paper)
                                if self.debug_mode:
                                    print(f"[DEBUG] 参考论文信息: title={paper.title[:50] if paper.title else 'N/A'}...")
                        
                        if self.debug_mode:
                            print(f"[DEBUG] 找到 {len(referenced_papers)} 篇被引用论文基本信息")
                        return referenced_papers
                    else:
                        response_text = await response.text()
                        if self.debug_mode:
                            print(f"[DEBUG] API请求失败，状态码: {response.status}")
                            print(f"[DEBUG] 错误响应: {response_text[:200]}...")
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"[DEBUG] 请求异常: {str(e)}")
                
        if self.debug_mode:
            print(f"[DEBUG] 返回空列表")
        return []

# Create a global server instance
server_instance = ArxivMCPServer()

# Register the tool with FastMCP
@mcp.tool()
async def analyze_arxiv_citations(
    arxiv_url: str,
    debug: bool = False # debug参数
) -> Dict[str, Any]:
    server_instance.debug_mode = debug # 根据参数设置debug模式
    """获取arxiv文章的引用关系和相关论文信息"""
    return await server_instance.analyze_paper_citations(arxiv_url)

if __name__ == "__main__":
    # Run the FastMCP server
    mcp.run()