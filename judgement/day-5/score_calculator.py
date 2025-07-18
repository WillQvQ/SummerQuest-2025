#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评分计算器 - 统计 grading_results_0717.md 文件中每个学生的得分

功能:
- 读取评分文件
- 统计每个学生的 hw5_1 和 hw5_2 得分
- 以 Markdown 表格格式输出结果
- 将结果写入飞书多维表格
"""

import re
import argparse
import os
import json
import time
import threading
import webbrowser
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import urlparse, parse_qs, urlencode
from urllib import request
from http.server import HTTPServer, BaseHTTPRequestHandler

# 飞书相关常量
FEISHU_HOST = "https://open.feishu.cn"
REDIRECT_URI = "http://localhost:8080/callback"
MAX_OPS = 500
WAITING_TIME = 0.1

class AuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/callback'):
            query = parse_qs(self.path.split('?')[1] if '?' in self.path else '')
            if 'code' in query:
                self.server.auth_code = query['code'][0]
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<html><body><h1>Authorization successful! You can close this window.</h1></body></html>')
            else:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<html><body><h1>Authorization failed!</h1></body></html>')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # 禁用日志输出

class SimpleLarkAuth:
    """飞书用户授权管理类"""
    
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.token_file = f"feishu_user_token_{app_id}.json"
        self.token_info = self._load_token_from_file()
    
    def _load_token_from_file(self) -> Optional[Dict[str, Any]]:
        """从文件加载token信息"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载token文件失败: {e}")
        return None
    
    def _save_token_to_file(self, token_info: Dict[str, Any]):
        """保存token信息到文件"""
        try:
            with open(self.token_file, 'w', encoding='utf-8') as f:
                json.dump(token_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存token文件失败: {e}")
    
    def _oauth_flow(self) -> Optional[str]:
        """执行OAuth授权流程"""
        # 构建授权URL
        auth_url = f"{FEISHU_HOST}/open-apis/authen/v1/index?app_id={self.app_id}&redirect_uri={REDIRECT_URI}&response_type=code&scope=offline_access bitable:app"
        
        print(f"请在浏览器中完成授权: `{auth_url}`")
        
        # 启动本地服务器接收回调
        server = HTTPServer(('localhost', 8080), AuthCallbackHandler)
        server.auth_code = None
        
        # 在新线程中启动服务器
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # 打开浏览器
        webbrowser.open(auth_url)
        
        print("等待用户授权中...")
        
        # 等待授权码
        timeout = 300  # 5分钟超时
        start_time = time.time()
        while server.auth_code is None and time.time() - start_time < timeout:
            time.sleep(1)
        
        server.shutdown()
        
        if server.auth_code:
            return self._get_auth_code(server.auth_code)
        else:
            print("授权超时或失败")
            return None
    
    def _get_auth_code(self, code: str) -> Optional[str]:
        """使用授权码获取访问token"""
        url = f"{FEISHU_HOST}/open-apis/authen/v1/access_token"
        headers = {
            "Authorization": f"Bearer {self._get_app_access_token()}",
            "Content-Type": "application/json; charset=utf-8"
        }
        req_body = {
            "grant_type": "authorization_code",
            "code": code
        }
        
        try:
            data = bytes(json.dumps(req_body), encoding='utf8')
            req = request.Request(url=url, data=data, headers=headers)
            response = request.urlopen(req)
            rsp_body = response.read().decode('utf-8')
            result = json.loads(rsp_body)
            
            if result.get("code") == 0:
                token_info = {
                    "access_token": result["data"]["access_token"],
                    "refresh_token": result["data"]["refresh_token"],
                    "expires_in": result["data"]["expires_in"],
                    "created_at": time.time()
                }
                self._save_token_to_file(token_info)
                return result["data"]["access_token"]
            else:
                print(f"OAuth失败: {result.get('msg', '未知错误')}")
                return None
        except Exception as e:
            print(f"OAuth请求失败: {e}")
            return None
    
    def _get_app_access_token(self) -> str:
        """获取应用访问token"""
        url = f"{FEISHU_HOST}/open-apis/auth/v3/app_access_token/internal"
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }
        req_body = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        try:
            data = bytes(json.dumps(req_body), encoding='utf8')
            req = request.Request(url=url, data=data, headers=headers)
            response = request.urlopen(req)
            rsp_body = response.read().decode('utf-8')
            result = json.loads(rsp_body)
            
            if result.get("code") == 0:
                return result["app_access_token"]
            else:
                raise Exception(f"获取app_access_token失败: {result.get('msg', '未知错误')}")
        except Exception as e:
            raise Exception(f"请求app_access_token异常: {e}")
    
    def _refresh_token(self) -> Optional[str]:
        """刷新访问token"""
        if not self.token_info or "refresh_token" not in self.token_info:
            return None
        
        url = f"{FEISHU_HOST}/open-apis/authen/v1/refresh_access_token"
        headers = {
            "Authorization": f"Bearer {self._get_app_access_token()}",
            "Content-Type": "application/json; charset=utf-8"
        }
        req_body = {
            "grant_type": "refresh_token",
            "refresh_token": self.token_info["refresh_token"]
        }
        
        try:
            data = bytes(json.dumps(req_body), encoding='utf8')
            req = request.Request(url=url, data=data, headers=headers)
            response = request.urlopen(req)
            rsp_body = response.read().decode('utf-8')
            result = json.loads(rsp_body)
            
            if result.get("code") == 0:
                token_info = {
                    "access_token": result["data"]["access_token"],
                    "refresh_token": result["data"]["refresh_token"],
                    "expires_in": result["data"]["expires_in"],
                    "created_at": time.time()
                }
                self._save_token_to_file(token_info)
                return result["data"]["access_token"]
            else:
                print(f"刷新token失败: {result.get('msg', '未知错误')}")
                return None
        except Exception as e:
            print(f"刷新token请求失败: {e}")
            return None
    
    def get_token(self) -> str:
        """获取有效的访问token"""
        # 检查现有token是否有效
        if self.token_info:
            created_at = self.token_info.get("created_at", 0)
            expires_in = self.token_info.get("expires_in", 0)
            if time.time() - created_at < expires_in - 300:  # 提前5分钟刷新
                return self.token_info["access_token"]
            
            # 尝试刷新token
            refreshed_token = self._refresh_token()
            if refreshed_token:
                self.token_info = self._load_token_from_file()
                return refreshed_token
        
        # 执行完整的OAuth流程
        token = self._oauth_flow()
        if token:
            self.token_info = self._load_token_from_file()
            return token
        
        raise Exception("无法获取有效的访问token")


class SimpleLark:
    """飞书简化操作类，支持读取和写入多维表格功能"""
    
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
    
    def batch_update_records(self, app_token: str, table_id: str, records: List[Dict[str, Any]]):
        """批量更新多条记录"""
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update"
        req_body = {"records": records}
        rsp_dict = self.post_req(url, req_body=req_body)
        if rsp_dict.get("code", -1) == 0:
            print(f"✅ 成功批量更新 {len(records)} 条记录")
            return len(records)
        else:
            print(f"❌ 批量更新失败: {rsp_dict.get('msg', '未知错误')}") 
            return 0
    
    def create_records(self, app_token: str, table_id: str, records: List[Dict[str, Any]]):
        """批量创建记录"""
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_create"
        req_body = {"records": records}
        rsp_dict = self.post_req(url, req_body=req_body)
        if rsp_dict.get("code", -1) == 0:
            print(f"✅ 成功创建 {len(records)} 条记录")
            return True
        else:
            print(f"❌ 创建记录失败: {rsp_dict.get('msg', '未知错误')}")
            return False


def calculate_scores(file_path):
    """Parses the grading file and calculates scores for each student."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

    scores = {}
    current_student = None
    current_section = None

    for line in lines:
        # Match student names, which are level 3 markdown headers
        match = re.match(r'^###\s+(.+)', line)
        if match:
            current_student = match.group(1).strip()
            if current_student not in scores:
                scores[current_student] = {'hw5_1': 0, 'hw5_2': 0}
            current_section = None
            continue

        if not current_student:
            continue

        # Determine the current scoring section (hw5_1 or hw5_2)
        if 'hw5_1 评分' in line:
            current_section = 'hw5_1'
            continue
        elif 'hw5_2 评分' in line:
            current_section = 'hw5_2'
            continue
        
        # Count the checkmarks (✓) to determine the raw score
        if current_section and '✓' in line:
            scores[current_student][current_section] += 1

    # Convert the raw score for hw5_1 to the final score (floor division by 2)
    for student, score_data in scores.items():
        score_data['hw5_1_converted'] = score_data['hw5_1'] // 2

    return scores

def print_scores_as_markdown(student_scores):
    """Prints the student scores in a markdown table format."""
    if not student_scores:
        return

    print('| 学生姓名 | hw5_1得分 | hw5_2得分 | 总分 |')
    print('|---------|----------|----------|------|')

    # Sort students by total score in descending order
    sorted_scores = sorted(student_scores.items(), key=lambda item: (item[1]['hw5_1_converted'] + item[1]['hw5_2']), reverse=True)

    for student, scores in sorted_scores:
        hw5_1_score = f"{scores['hw5_1_converted']}/5"
        hw5_2_score = f"{scores['hw5_2']}/5"
        total_score = scores['hw5_1_converted'] + scores['hw5_2']
        print(f'| {student} | {hw5_1_score} | {hw5_2_score} | {total_score} |')

def write_to_feishu(scores: Dict[str, Dict[str, int]]):
    """将分数写入飞书多维表格"""
    # 固定的表格URL
    TABLE_URL = "https://fudan-nlp.feishu.cn/base/KH8obWHvqam2Y4sXGGuct2HFnEb?table=tblEWELbFTgWi3yY&view=vewq2qW6vT"
    
    # 从环境变量读取飞书应用信息
    app_id = os.getenv('FEISHU_APP_ID')
    app_secret = os.getenv('FEISHU_APP_SECRET')
    
    if not app_id or not app_secret:
        print("❌ 请设置环境变量 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        return
    
    try:
        # 初始化飞书客户端
        lark = SimpleLark(app_id, app_secret)
        
        # 解析表格URL
        parsed_url = urlparse(TABLE_URL)
        path_parts = parsed_url.path.split('/')
        app_token = path_parts[2]
        table_id = parsed_url.query.split('&')[0].split('=')[1]
        
        # 获取所有记录
        records = lark.get_records(app_token, table_id)
        
        # 收集飞书表格中的姓名集合
        feishu_names = {record['fields'].get('姓名', '') for record in records}
        
        # 收集需要更新的记录
        updates = []
        missing_students = []
        
        print(f"\n📋 准备更新飞书表格...")
        print(f"📊 需要处理的学生数量: {len(scores)}")
        
        for student, score_data in scores.items():
            if student not in feishu_names:
                missing_students.append(student)
                print(f"⚠️ 学生 {student} 在飞书表格中不存在")
                continue
            
            hw5_1_score = score_data['hw5_1_converted']
            hw5_2_score = score_data['hw5_2']
            
            # 查找对应的记录并准备更新
            for record in records:
                if record['fields'].get('姓名', '') == student:
                    updates.append({
                        'record_id': record['record_id'],
                        'fields': {
                            'Day-5-hw1': hw5_1_score,
                            'Day-5-hw2': hw5_2_score
                        }
                    })
                    print(f"✅ 准备更新学生 {student}: Day-5-hw1={hw5_1_score}, Day-5-hw2={hw5_2_score}")
                    break
        
        # 批量更新记录
        if updates:
            print(f"\n🔄 正在批量更新 {len(updates)} 条记录...")
            lark.batch_update_records(app_token, table_id, updates)
            print(f"✅ 飞书表格更新完成!")
        else:
            print(f"\n📝 没有需要更新的记录")
        
        # 打印统计信息
        print(f"\n📈 更新统计:")
        print(f"   - 成功更新: {len(updates)} 人")
        print(f"   - 表格中不存在: {len(missing_students)} 人")
        
        # 打印缺失的学生姓名
        if missing_students:
            print("\n⚠️ 以下学生在飞书表格中不存在:")
            for student in missing_students:
                print(f"   - {student}")
            
    except Exception as e:
        print(f"\n❌ 飞书操作失败: {str(e)}")


def main():
    """Main function to parse arguments and run the score calculation."""
    parser = argparse.ArgumentParser(description='Calculate student scores from a markdown file.')
    parser.add_argument('file_path', help='The path to the grading results markdown file.')
    parser.add_argument('--write-feishu', action='store_true', help='将结果写入飞书表格')
    
    args = parser.parse_args()

    try:
        student_scores = calculate_scores(args.file_path)
        print_scores_as_markdown(student_scores)
        
        # 如果指定了写入飞书，则写入飞书表格
        if args.write_feishu:
            print("\n🔄 正在写入飞书表格...")
            write_to_feishu(student_scores)
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {args.file_path}")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")

if __name__ == '__main__':
    main()