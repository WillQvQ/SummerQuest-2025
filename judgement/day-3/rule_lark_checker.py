#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
day3_lark_checker.py - Day-3作业检查脚本

检查项目：
1. hw3_1.json: 检查special_tokens的id是否出现在每个token_ids序列中
2. hw3_2.json: 检查Output字段格式（基于output_checker.py）
   - 是否包含 think 部分
   - 除think外展示给用户的部分，是否含有特殊词符 <|EDIT|> 和 <|AGENT|> 之一
   - <|AGENT|> 后是否正确调用函数 python
   - <|EDIT|> 后是否调用函数 editor

评分规则：
- Day-3-hw1: 0分或2分（hw3_1.json格式正确且special_tokens的id都出现在token_ids中）
- Day-3-hw2: 0-8分（hw3_2.json中每个正确的Output得1分，最多8分）
"""

import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Dict, List, Any, Tuple
from urllib import request
from urllib.parse import urlencode, urlparse, parse_qs
import requests
import re

WAITING_TIME = 0.01
MAX_OPS = 500
TABLE_URL = "https://fudan-nlp.feishu.cn/base/KH8obWHvqam2Y4sXGGuct2HFnEb?table=tblEWELbFTgWi3yY&view=vewq2qW6vT"

class SimpleLarkAuth:
    """飞书简化授权管理类，仅支持用户授权模式，支持token缓存和自动刷新"""
    
    def __init__(self, app_id: str, app_secret: str, redirect_uri: str = "http://localhost:8080/callback", token_file: str = "feishu_token.json"):
        self.APP_ID = app_id
        self.APP_SECRET = app_secret
        self.REDIRECT_URI = redirect_uri
        self.TOKEN_FILE = token_file
        self._current_token: Optional[str] = None
        self._token_expire_time: float = 0
    
    def get_token(self, force_refresh: bool = False) -> str:
        """获取用户访问令牌，支持自动缓存和刷新"""
        now_time = time.time()
        
        # 如果内存中有有效token，直接返回
        if not force_refresh and self._current_token and now_time < self._token_expire_time:
            return self._current_token
        
        # 尝试从本地文件加载token
        token_data = self._load_token_from_file()
        
        if token_data and not force_refresh:
            # 检查access_token是否还有效
            if now_time < token_data["access_token_expires_at"]:
                self._current_token = token_data["access_token"]
                self._token_expire_time = token_data["access_token_expires_at"]
                if not self._current_token:
                    raise Exception("缓存的access_token为空")
                return self._current_token
            
            # access_token过期，检查refresh_token是否还有效
            elif now_time < token_data["refresh_token_expires_at"]:
                print("🔄 正在刷新access_token...")
                try:
                    new_token = self._refresh_access_token(token_data["refresh_token"])
                    return new_token
                except Exception as e:
                    print(f"⚠️ 刷新token失败: {str(e)}")
            
            # refresh_token也过期了
            else:
                print("⚠️ refresh_token也已过期，需要重新授权")
        
        # 如果没有有效缓存或刷新失败，进行完整的OAuth授权流程
        print("🔐 开始OAuth授权...")
        return self._do_full_oauth()
    
    def _load_token_from_file(self) -> Optional[Dict[str, Any]]:
        """从文件加载token信息"""
        if not os.path.exists(self.TOKEN_FILE):
            return None
        
        try:
            with open(self.TOKEN_FILE, "r", encoding="utf-8") as f:
                token_data = json.load(f)
            
            # 验证token数据完整性
            required_fields = ["access_token", "refresh_token", "access_token_expires_at", "refresh_token_expires_at"]
            if all(field in token_data for field in required_fields):
                return token_data
            else:
                print("⚠️ token文件格式不完整，将重新授权")
                return None
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 读取token文件失败: {str(e)}")
            return None
    
    def _save_token_to_file(self, access_token: str, refresh_token: str, expires_in: int, refresh_expires_in: int):
        """保存token信息到文件"""
        now_time = time.time()
        token_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "access_token_expires_at": now_time + expires_in - 300,  # 提前5分钟过期
            "refresh_token_expires_at": now_time + refresh_expires_in - 300,  # 提前5分钟过期
            "created_at": now_time,
            "updated_at": now_time
        }
        
        try:
            with open(self.TOKEN_FILE, "w", encoding="utf-8") as f:
                json.dump(token_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️ 保存token文件失败: {str(e)}")
        
        # 更新内存中的token
        self._current_token = access_token
        self._token_expire_time = token_data["access_token_expires_at"]
    
    def _refresh_access_token(self, refresh_token: str) -> str:
        """使用refresh_token刷新access_token"""
        url = "https://open.feishu.cn/open-apis/authen/v1/refresh_access_token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }
        
        try:
            response = requests.post(
                url, 
                json=data, 
                auth=(self.APP_ID, self.APP_SECRET),  # 使用Basic Auth
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("code", -1) != 0:
                raise Exception(f"刷新Token失败: {result.get('msg', '未知错误')}")
            
            # 刷新API直接返回token信息，不需要访问data字段
            if "access_token" not in result:
                raise Exception(f"API响应中缺少access_token")
            
            # 保存新的token信息
            self._save_token_to_file(
                access_token=result["access_token"],
                refresh_token=result.get("refresh_token", refresh_token),  # 有些情况下不返回新的refresh_token
                expires_in=result.get("expires_in", 7200),
                refresh_expires_in=result.get("refresh_token_expires_in", 604800)
            )
            
            print("✅ Token刷新成功")
            if not self._current_token:
                raise Exception("刷新后的token为空")
            return self._current_token
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"刷新Token网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"解析刷新Token响应失败: {str(e)}")
    
    def _do_full_oauth(self) -> str:
        """执行完整的OAuth授权流程"""
        auth_code = self._get_auth_code()
        if not auth_code:
            raise Exception("未获取到授权码")
        
        url = "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": self.APP_ID,
            "client_secret": self.APP_SECRET,
            "redirect_uri": self.REDIRECT_URI
        }
        
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("code", -1) != 0:
                raise Exception(f"获取Token失败: {result.get('msg', '未知错误')}")
            
            if "access_token" not in result:
                raise Exception(f"API响应中缺少access_token")
            
            # 保存token信息
            self._save_token_to_file(
                access_token=result["access_token"],
                refresh_token=result.get("refresh_token", ""),
                expires_in=result.get("expires_in", 7200),
                refresh_expires_in=result.get("refresh_token_expires_in", 604800)
            )
            
            print("✅ OAuth授权成功")
            if not self._current_token:
                raise Exception("OAuth后的token为空")
            return self._current_token
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"OAuth请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"解析OAuth响应失败: {str(e)}")
    
    def clear_cache(self):
        """清除本地token缓存"""
        if os.path.exists(self.TOKEN_FILE):
            try:
                os.remove(self.TOKEN_FILE)
                print(f"🗑️ 已清除token缓存文件: {self.TOKEN_FILE}")
            except OSError as e:
                print(f"⚠️ 删除token文件失败: {str(e)}")
        
        self._current_token = None
        self._token_expire_time = 0
    
    def get_token_info(self) -> Optional[Dict[str, Any]]:
        """获取当前token信息（用于调试）"""
        token_data = self._load_token_from_file()
        if token_data:
            now = time.time()
            token_data["access_token_valid"] = now < token_data["access_token_expires_at"]
            token_data["refresh_token_valid"] = now < token_data["refresh_token_expires_at"]
            token_data["access_token_expires_in_seconds"] = max(0, token_data["access_token_expires_at"] - now)
            token_data["refresh_token_expires_in_seconds"] = max(0, token_data["refresh_token_expires_at"] - now)
        return token_data
    
    def _get_auth_code(self) -> Optional[str]:
        """获取用户授权码"""
        class AuthServer(HTTPServer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.auth_code: Optional[str] = None
        
        class AuthHandler(BaseHTTPRequestHandler):
            server: AuthServer  # 类型注解
            
            def do_GET(self):
                query = urlparse(self.path).query
                params = parse_qs(query)
                
                if "code" in params:
                    self.server.auth_code = params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<h1>Success! You can close this page.</h1>")
                else:
                    self.send_response(400)
                    self.end_headers()
                
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            
            def log_message(self, format, *args):
                pass
        
        server = AuthServer(("localhost", 8080), AuthHandler)
        
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        auth_url = (
            f"https://open.feishu.cn/open-apis/authen/v1/index?"
            f"app_id={self.APP_ID}&redirect_uri={self.REDIRECT_URI}"
            f"&response_type=code&scope=offline_access bitable:app"
        )
        print(f"请在浏览器中完成授权: {auth_url}")
        webbrowser.open(auth_url)
        
        print("等待用户授权中...")
        timeout = 300
        start_time = time.time()
        
        while server.auth_code is None and server_thread.is_alive():
            if time.time() - start_time > timeout:
                print("获取授权码超时，请重试")
                return None
            time.sleep(0.5)
        
        return server.auth_code


class SimpleLark:
    """飞书简化操作类，支持读取多维表格功能"""
    
    def __init__(self, app_id: str, app_secret: str, bitable_url: Optional[str] = None):
        self.auth = SimpleLarkAuth(app_id, app_secret)
        self._bitable_dict: Dict[str, Dict[str, str]] = {}
        if bitable_url:
            self.add_bitable("default", bitable_url)
    
    def _post_req(self, url: str, headers: Dict[str, str], req_body: Dict[str, Any], param: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if param is not None:
            url = url + '?' + urlencode(param)
        try:
            data = bytes(json.dumps(req_body), encoding='utf8')
            req = request.Request(url=url, data=data, headers=headers, method='POST')
            response = request.urlopen(req)
            rsp_body = response.read().decode('utf-8')
            result = json.loads(rsp_body)
            return result
        except Exception as e:
            print(f"❌ POST请求失败: {str(e)}")
            return {"code": -1, "msg": f"请求失败: {str(e)}"}
    
    def _get_req(self, url: str, param: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if param is not None:
            url = url + '?' + urlencode(param)
        try:
            req = request.Request(url=url, method='GET')
            req.add_header('Authorization', 'Bearer {}'.format(self.auth.get_token()))
            response = request.urlopen(req)
            rsp_body = response.read().decode('utf-8')
            result = json.loads(rsp_body)
            return result
        except Exception as e:
            print(f"❌ GET请求失败: {str(e)}")
            return {"code": -1, "msg": f"请求失败: {str(e)}"}
    
    def post_req(self, url: str, headers: Optional[Dict[str, str]] = None, req_body: Optional[Dict[str, Any]] = None, param: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if headers is None:
            headers = {}
        if req_body is None:
            req_body = {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.auth.get_token()
        return self._post_req(url, headers, req_body, param)
    
    def get_req(self, url: str, headers: Optional[Dict[str, str]] = None, param: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if headers is None:
            headers = {}
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.auth.get_token()
        return self._get_req(url, param)
    
    def add_bitable(self, table_name: str, link: str):
        """添加多维表格配置"""
        if table_name in self._bitable_dict:
            print("Error! Table name {} has been saved in config.".format(table_name))
            return
            
        link_end = link.split("/")[-1]
        app_token = link_end.split("?")[0]
        params = link_end.split("?")[-1].split('&')
        table_id = ""
        
        for param in params:
            try:
                if param.split("=")[0] == 'table':
                    table_id = param.split("=")[1]
            except IndexError:
                pass
                
        if table_id == "":
            print("Error! Table id is not been found")
            return
            
        self._bitable_dict[table_name] = {
            "app_token": app_token,
            "table_id": table_id
        }
    
    def bitable(self, table_name: str = "default") -> tuple[str, str]:
        """获取多维表格配置"""
        if table_name not in self._bitable_dict:
            raise KeyError("未找到名为{}的多维表格".format(table_name))
        item = self._bitable_dict[table_name]
        return item["app_token"], item["table_id"]
    
    def bitable_list(self, app_token: str, table_id: str, filter_dict: Optional[Dict[str, str]] = None, page_token: str = "") -> tuple[List[Dict[str, Any]], Optional[str]]:
        """分页获取多维表格记录"""
        if filter_dict is None:
            filter_dict = {}

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
        param = {
            "page_size": str(MAX_OPS),
            **filter_dict
        }
        if page_token != "":
            param["page_token"] = page_token
            
        rsp_dict = self.get_req(url, param=param)

        if rsp_dict.get("code", -1) == 0:
            # 安全访问data字段
            if "data" not in rsp_dict:
                print(f"❌ API响应中缺少data字段")
                return [], None
            
            data = rsp_dict["data"]
            has_more = data.get("has_more", False)
            next_page_token = data.get("page_token", "") if has_more else None
            return data.get("items", []), next_page_token
        else:
            print(f"❌ 获取记录失败: {rsp_dict.get('msg', '未知错误')}")
            return [], None
    
    def get_records(self, app_token: str, table_id: str) -> List[Dict[str, Any]]:
        """获取所有记录"""
        all_records = []
        page_token = ""
        
        while True:
            records, next_page_token = self.bitable_list(
                app_token, 
                table_id, 
                page_token=page_token
            )
            
            all_records.extend(records)
            
            if next_page_token is None:
                break
            page_token = next_page_token
            
            # 防止无限循环
            time.sleep(WAITING_TIME)
        
        print(f"✅ 成功获取 {len(all_records)} 条记录")
        return all_records
    
    def _put_req(self, url: str, headers: Dict[str, str], req_body: Dict[str, Any], param: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if param is not None:
            url = url + '?' + urlencode(param)
        try:
            data = bytes(json.dumps(req_body), encoding='utf8')
            req = request.Request(url=url, data=data, headers=headers, method='PUT')
            response = request.urlopen(req)
            rsp_body = response.read().decode('utf-8')
            result = json.loads(rsp_body)
            return result
        except Exception as e:
            print(f"❌ PUT请求失败: {str(e)}")
            return {"code": -1, "msg": f"请求失败: {str(e)}"}

    def put_req(self, url: str, headers: Optional[Dict[str, str]] = None, req_body: Optional[Dict[str, Any]] = None, param: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if headers is None:
            headers = {}
        if req_body is None:
            req_body = {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.auth.get_token()
        return self._put_req(url, headers, req_body, param)
    
    def update_record(self, app_token: str, table_id: str, record_id: str, fields: Dict[str, Any]):
        """更新单条记录"""
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}"
        req_body = {"fields": fields}
        rsp_dict = self.put_req(url, req_body=req_body)
        if rsp_dict.get("code", -1) == 0:
            print(f"✅ 成功更新记录 {record_id}")
        else:
            print(f"❌ 更新记录失败: {rsp_dict.get('msg', '未知错误')}")

    def batch_update_records(self, app_token: str, table_id: str, records: List[Dict[str, Any]]):
        """批量更新多条记录"""
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update"
        req_body = {"records": records}
        rsp_dict = self.post_req(url, req_body=req_body)
        if rsp_dict.get("code", -1) == 0:
            print(f"✅ 成功批量更新 {len(records)} 条记录")
        else:
            print(f"❌ 批量更新失败: {rsp_dict.get('msg', '未知错误')}")


# ==================== Day-3 作业检查逻辑 ====================

def check_hw3_1_json(file_path: str) -> Tuple[bool, str]:
    """
    检查 hw3_1.json 文件
    
    Args:
        file_path: hw3_1.json 文件路径
        
    Returns:
        tuple: (是否通过, 详细信息)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return False, f"文件未找到: {file_path}"
    except json.JSONDecodeError as e:
        return False, f"JSON 解析错误: {e}"
    
    # 检查基本结构
    if not isinstance(data, dict):
        return False, "数据格式错误：应该是一个字典"
    
    if "special_tokens" not in data:
        return False, "缺少 special_tokens 字段"
    
    if "tasks" not in data:
        return False, "缺少 tasks 字段"
    
    special_tokens = data["special_tokens"]
    tasks = data["tasks"]
    
    if not isinstance(special_tokens, list):
        return False, "special_tokens 应该是一个列表"
    
    if not isinstance(tasks, list):
        return False, "tasks 应该是一个列表"
    
    # 提取 special_tokens 的 id
    special_token_ids = set()
    for token in special_tokens:
        if not isinstance(token, dict) or "id" not in token:
            return False, "special_tokens 中的项目格式错误"
        special_token_ids.add(token["id"])
    
    if not special_token_ids:
        return False, "special_tokens 为空"
    
    # 检查每个 task 的 token_ids 中是否包含至少一个 special_token_id
    missing_details = []
    for i, task in enumerate(tasks):
        if not isinstance(task, dict) or "token_ids" not in task:
            return False, f"tasks[{i}] 格式错误：缺少 token_ids 字段"
        
        token_ids = task["token_ids"]
        if not isinstance(token_ids, list):
            return False, f"tasks[{i}] 的 token_ids 应该是一个列表"
        
        token_ids_set = set(token_ids)
        # 检查是否包含至少一个special_token_id
        has_special_token = bool(special_token_ids & token_ids_set)
        
        if not has_special_token:
            missing_details.append(f"tasks[{i}] 不包含任何 special_token_ids: {special_token_ids}")
    
    if missing_details:
        return False, "\n".join(missing_details)
    
    return True, f"✅ hw3_1.json 检查通过，共 {len(tasks)} 个任务，所有任务的 token_ids 都包含至少一个 special_token_id"


def extract_think_content(output: str) -> Tuple[str, str]:
    """
    提取 think 部分和非 think 部分的内容
    
    Args:
        output: 完整的输出字符串
        
    Returns:
        tuple: (think_content, non_think_content)
    """
    # 匹配 <think>...</think> 标签
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, output, re.DOTALL)
    
    # 提取 think 内容
    think_content = '\n'.join(think_matches) if think_matches else ''
    
    # 移除 think 部分，得到非 think 内容
    non_think_content = re.sub(think_pattern, '', output, flags=re.DOTALL).strip()
    
    return think_content, non_think_content


def check_special_markers(non_think_content: str) -> Tuple[bool, str]:
    """
    检查是否包含特殊词符 <|EDIT|> 或 <|AGENT|>
    
    Args:
        non_think_content: 非think部分的内容
        
    Returns:
        tuple: (has_marker, marker_type)
    """
    if '<|EDIT|>' in non_think_content:
        return True, 'EDIT'
    elif '<|AGENT|>' in non_think_content:
        return True, 'AGENT'
    else:
        return False, 'NONE'


def check_function_call(content: str, expected_function: str) -> Tuple[bool, str]:
    """
    检查是否正确调用了指定的函数
    
    Args:
        content: 要检查的内容
        expected_function: 期望的函数名 ('python' 或 'editor')
        
    Returns:
        tuple: (has_correct_call, details)
    """
    # 匹配 JSON 格式的函数调用
    function_call_pattern = r'{\s*"name"\s*:\s*"([^"]+)"'
    matches = re.findall(function_call_pattern, content)
    
    if matches:
        for match in matches:
            if match == expected_function:
                return True, f"找到正确的{expected_function}函数调用"
        return False, f"找到函数调用但不是{expected_function}: {matches}"
    else:
        return False, f"未找到{expected_function}函数调用"


def check_single_output(output: str, index: int) -> Dict:
    """
    检查单个输出项
    
    Args:
        output: 输出字符串
        index: 项目索引
        
    Returns:
        dict: 检查结果
    """
    result = {
        'index': index,
        'has_think': False,
        'has_special_marker': False,
        'marker_type': 'NONE',
        'correct_function_call': False,
        'function_call_details': '',
        'issues': []
    }
    
    # 1. 检查是否包含 think 部分
    think_content, non_think_content = extract_think_content(output)
    result['has_think'] = bool(think_content.strip())
    
    if not result['has_think']:
        result['issues'].append('缺少 <think> 部分')
    
    # 2. 检查特殊词符
    has_marker, marker_type = check_special_markers(non_think_content)
    result['has_special_marker'] = has_marker
    result['marker_type'] = marker_type
    
    if not has_marker:
        result['issues'].append('缺少特殊词符 <|EDIT|> 或 <|AGENT|>')
    
    # 3. 根据标记类型检查函数调用
    if marker_type == 'AGENT':
        # 检查是否调用了 python 函数
        has_correct_call, details = check_function_call(non_think_content, 'python')
        result['correct_function_call'] = has_correct_call
        result['function_call_details'] = details
        
        if not has_correct_call:
            result['issues'].append('<|AGENT|> 后未正确调用 python 函数')
            
    elif marker_type == 'EDIT':
        # 检查是否调用了 editor 函数
        has_correct_call, details = check_function_call(non_think_content, 'editor')
        result['correct_function_call'] = has_correct_call
        result['function_call_details'] = details
        
        if not has_correct_call:
            result['issues'].append('<|EDIT|> 后未正确调用 editor 函数')
    
    return result


def check_hw3_2_json(file_path: str) -> Tuple[int, str]:
    """
    检查 hw3_2.json 文件
    
    Args:
        file_path: hw3_2.json 文件路径
        
    Returns:
        tuple: (得分, 详细信息)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0, f"文件未找到: {file_path}"
    except json.JSONDecodeError as e:
        return 0, f"JSON 解析错误: {e}"
    
    if not isinstance(data, list):
        return 0, "数据格式错误：应该是一个列表"
    
    passed_items = 0
    details = []
    
    for i, item in enumerate(data):
        if not isinstance(item, dict) or 'Output' not in item:
            details.append(f"项目 {i}: ❌ 格式错误：缺少 Output 字段")
            continue
        
        output = item['Output']
        check_result = check_single_output(output, i)
        
        if not check_result['issues']:
            passed_items += 1
            details.append(f"项目 {i}: ✅ 通过所有检查")
        else:
            issues_str = ', '.join(check_result['issues'])
            details.append(f"项目 {i}: ❌ {issues_str}")
    
    # 最多8分
    score = min(passed_items, 8)
    
    summary = f"✅ hw3_2.json 检查完成，共 {len(data)} 个项目，通过 {passed_items} 个，得分 {score}/8\n\n" + "\n".join(details)
    
    return score, summary


def main():
    """主函数"""
    app_id = os.getenv('FEISHU_APP_ID')
    app_secret = os.getenv('FEISHU_APP_SECRET')
    if not app_id or not app_secret:
        print("Please set FEISHU_APP_ID and FEISHU_APP_SECRET environment variables.")
        return

    lark = SimpleLark(app_id, app_secret)

    # Parse the table URL
    parsed_url = urlparse(TABLE_URL)
    path_parts = parsed_url.path.split('/')
    app_id_from_url = path_parts[2]
    table_id = parsed_url.query.split('&')[0].split('=')[1]

    # 获取所有记录
    records = lark.get_records(app_id_from_url, table_id)

    # 收集飞书表格中的姓名集合
    feishu_names = {record['fields'].get('姓名', '') for record in records}

    # 计算分数
    BASE_PATH = '../submission'
    scores = {}  # {student_name: {'hw1': score, 'hw2': score}}
    
    for student_dir in os.listdir(BASE_PATH):
        student_path = os.path.join(BASE_PATH, student_dir)
        if os.path.isdir(student_path):
            print(f"\n🔍 开始检查学生 {student_dir} 的Day-3作业...")
            day3_path = os.path.join(student_path, 'day-3')
            
            if not os.path.isdir(day3_path):
                print(f"❌ 学生 {student_dir} 没有day-3目录，跳过")
                continue
            
            # 检查该学生在飞书表格中是否已有数据
            existing_hw1_score = None
            existing_hw2_score = None
            for record in records:
                if record['fields'].get('姓名', '') == student_dir:
                    existing_hw1_score = record['fields'].get('Day-3-hw1')
                    existing_hw2_score = record['fields'].get('Day-3-hw2')
                    break
            
            if (existing_hw1_score is not None and existing_hw1_score != '' and 
                existing_hw2_score is not None and existing_hw2_score != ''):
                print(f"⏭️ 学生 {student_dir} 在表格中已有数据 (hw1: {existing_hw1_score}, hw2: {existing_hw2_score})，跳过重复判断")
                scores[student_dir] = {'hw1': existing_hw1_score, 'hw2': existing_hw2_score}
                continue
            
            # 检查 hw3_1.json
            hw3_1_path = os.path.join(day3_path, 'hw3_1.json')
            hw1_score = 0
            if os.path.exists(hw3_1_path):
                print(f"\n📄 检查 hw3_1.json...")
                passed, details = check_hw3_1_json(hw3_1_path)
                if passed:
                    hw1_score = 2
                    print(f"✅ hw3_1.json 检查通过，得分: 2/2")
                else:
                    print(f"❌ hw3_1.json 检查失败: {details}")
                    print(f"❌ hw3_1.json 得分: 0/2")
            else:
                print(f"❌ 未找到 hw3_1.json 文件，得分: 0/2")
            
            # 检查 hw3_2.json
            hw3_2_path = os.path.join(day3_path, 'hw3_2.json')
            hw2_score = 0
            if os.path.exists(hw3_2_path):
                print(f"\n📄 检查 hw3_2.json...")
                hw2_score, details = check_hw3_2_json(hw3_2_path)
                print(f"📊 hw3_2.json 得分: {hw2_score}/8")
                print(details)
            else:
                print(f"❌ 未找到 hw3_2.json 文件，得分: 0/8")
            
            scores[student_dir] = {'hw1': hw1_score, 'hw2': hw2_score}
            print(f"\n📈 学生 {student_dir} 总分: Day-3-hw1={hw1_score}/2, Day-3-hw2={hw2_score}/8")

    # 收集需要更新的记录
    updates = []
    missing_students = []
    skipped_students = []
    
    print(f"\n📋 准备更新飞书表格...")
    print(f"📊 需要处理的学生数量: {len(scores)}")
    
    for student, student_scores in scores.items():
        if student not in feishu_names:
            missing_students.append(student)
            print(f"⚠️ 学生 {student} 在飞书表格中不存在")
            continue
            
        # 检查是否需要更新（避免重复更新已有数据）
        needs_update = True
        for record in records:
            if record['fields'].get('姓名', '') == student:
                existing_hw1 = record['fields'].get('Day-3-hw1')
                existing_hw2 = record['fields'].get('Day-3-hw2')
                if (existing_hw1 is not None and existing_hw1 != '' and existing_hw1 == student_scores['hw1'] and
                    existing_hw2 is not None and existing_hw2 != '' and existing_hw2 == student_scores['hw2']):
                    skipped_students.append(student)
                    needs_update = False
                    print(f"⏭️ 学生 {student} 的数据无需更新 (hw1: {existing_hw1}, hw2: {existing_hw2})")
                    break
                    
        if needs_update:
            for record in records:
                if record['fields'].get('姓名', '') == student:
                    updates.append({
                        'record_id': record['record_id'],
                        'fields': {
                            'Day-3-hw1': student_scores['hw1'],
                            'Day-3-hw2': student_scores['hw2']
                        }
                    })
                    print(f"✅ 准备更新学生 {student}: Day-3-hw1={student_scores['hw1']}, Day-3-hw2={student_scores['hw2']}")
                    break

    # 批量更新记录
    if updates:
        print(f"\n🔄 正在批量更新 {len(updates)} 条记录...")
        lark.batch_update_records(app_id_from_url, table_id, updates)
        print(f"✅ 飞书表格更新完成!")
    else:
        print(f"\n📝 没有需要更新的记录")

    # 打印统计信息
    print(f"\n📈 更新统计:")
    print(f"   - 成功更新: {len(updates)} 人")
    print(f"   - 跳过更新: {len(skipped_students)} 人")
    print(f"   - 表格中不存在: {len(missing_students)} 人")
    
    # 打印缺失的学生姓名
    if missing_students:
        print("\n⚠️ 以下学生在飞书表格中不存在:")
        for student in missing_students:
            print(f"   - {student}")


if __name__ == '__main__':
    main()