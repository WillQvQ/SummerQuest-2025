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

WAITING_TIME = 0.01
MAX_OPS = 500

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
    
    def bitable(self, table_name: str = "default") -> Tuple[str, str]:
        """获取多维表格配置"""
        if table_name not in self._bitable_dict:
            raise KeyError("未找到名为{}的多维表格".format(table_name))
        item = self._bitable_dict[table_name]
        return item["app_token"], item["table_id"]
    
    def bitable_list(self, app_token: str, table_id: str, filter_dict: Optional[Dict[str, str]] = None, page_token: str = "") -> Tuple[List[Dict[str, Any]], Optional[str]]:
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
    
    def get_records(self, table_name: str = "default") -> List[Dict[str, Any]]:
        """获取所有记录"""
        app_token, table_id = self.bitable(table_name)
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
    
    def get_filtered_records(self, table_name: str = "default", field_name: str = "", field_value: str = "") -> List[Dict[str, Any]]:
        """获取筛选后的记录
        
        注意：飞书的筛选语法比较复杂，这里先用简单的客户端筛选
        """
        if field_name and field_value:
            # 先获取所有记录，然后在客户端进行筛选
            all_records = self.get_records(table_name)
            filtered_records = []
            
            for record in all_records:
                fields = record.get("fields", {})
                field_val = fields.get(field_name, "")
                
                # 支持字符串包含匹配
                if isinstance(field_val, str) and field_value in field_val:
                    filtered_records.append(record)
                elif str(field_val) == field_value:
                    filtered_records.append(record)
            
            return filtered_records
        else:
            return self.get_records(table_name)


# 使用示例
if __name__ == "__main__":
    # 从环境变量获取配置信息
    app_id = os.environ.get("FEISHU_APP_ID", "cli_a8e08983d039900d")
    app_secret = os.environ.get("FEISHU_APP_SECRET", "fhyIeuzcvzY7sU91MmVEZeDJfQ8QM0cm")
    bitable_url = "https://fudan-nlp.feishu.cn/base/KH8obWHvqam2Y4sXGGuct2HFnEb?table=tbljlS1fS0UepxBn&view=vewCig26Kk"
    

    lark = SimpleLark(app_id, app_secret, bitable_url)
    
    print("=" * 50)
    print("查询主讲或助教中包含'刘智耿'的条目")
    print("=" * 50)
    
    try:
        all_records = lark.get_records()
        
        target_name = "刘智耿"
        matching_records = []
        
        for record in all_records:
            fields = record.get("fields", {})
            found = False
            
            main_lecturer = fields.get("主讲", "")
            if isinstance(main_lecturer, list):
                for item in main_lecturer:
                    if isinstance(item, dict) and 'name' in item:
                        if target_name in item['name']:
                            found = True
                            break
                    elif target_name in str(item):
                        found = True
                        break
            elif target_name in str(main_lecturer):
                found = True
            
            if not found:
                assistant = fields.get("助教", "")
                if isinstance(assistant, list):
                    for item in assistant:
                        if isinstance(item, dict) and 'name' in item:
                            if target_name in item['name']:
                                found = True
                                break
                        elif target_name in str(item):
                            found = True
                            break
                elif target_name in str(assistant):
                    found = True
            
            if found:
                matching_records.append(record)
        
        print(f"找到 {len(matching_records)} 条包含'{target_name}'的记录:")
        print()
        
        for i, record in enumerate(matching_records, 1):
            print(f"记录 {i}:")
            fields = record.get("fields", {})
            
            for field_name in ["课程", "主讲", "助教", "日期", "地点"]:
                if field_name in fields:
                    field_value = fields[field_name]
                    
                    # 格式化显示
                    if isinstance(field_value, list) and field_value:
                        if isinstance(field_value[0], dict) and 'name' in field_value[0]:
                            display_value = ", ".join([item.get('name', '') for item in field_value])
                        else:
                            display_value = ", ".join([str(item) for item in field_value])
                    else:
                        display_value = str(field_value)
                    
                    print(f"  {field_name}: {display_value}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 查询失败: {str(e)}")
    
    print("=" * 50)
    print("查询完成")
    print("=" * 50)