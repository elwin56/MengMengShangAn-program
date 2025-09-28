import os
import sqlite3
import datetime
import json
import re
import gradio as gr
import time
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType, ModelPlatformType
from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.utils import print_text_animated
from dotenv import load_dotenv
from camel.toolkits.mcp_toolkit import MCPToolkit

# 导入MCP相关模块
from lazyllm.tools import MCPClient
from lazyllm import OnlineChatModule

# 解决pandas版本兼容问题
try:
    import pandas as pd
    # 尝试设置选项，如果失败则忽略
    try:
        pd.set_option("future.no_silent_downcasting", True)
    except:
        pass
except ImportError:
    pd = None  # 标记pandas未导入

load_dotenv(dotenv_path='.env')

# 初始化MCP客户端和搜索工具
api_key = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
if not api_key:
    raise ValueError(
        "API key not found in environment variables. Please check your .env file."
    )

# 打印部分API密钥用于调试
print(f"Using API key: {api_key[:8]}...{api_key[-4:]}")

# 初始化MCP客户端，连接到Web搜索服务
saving_client = MCPClient(
    command_or_url="https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
    headers={
        "Authorization": f"Bearer {api_key}",
    },
    timeout=30,
)

# 获取MCP工具
tools = saving_client.get_tools()
search_tool = tools[0] if tools else None  # 获取web_search工具

# 打印可用的工具列表
print("Available tools:")
for i, tool in enumerate(tools):
    print(f"{i + 1}. Tool name: {tool.__name__}")

# 设置DashScope API密钥
os.environ['DASHSCOPE_API_KEY'] = api_key

def get_current_time():
    """获取当前本地时间，确保时区正确"""
    try:
        # 尝试使用本地时区
        return datetime.datetime.now()
    except:
        # 如果本地时区有问题，使用UTC时间
        return datetime.datetime.utcnow()

def get_current_time_str(format_str="%Y-%m-%d %H:%M:%S"):
    """获取当前时间字符串"""
    return get_current_time().strftime(format_str)

def get_current_time_iso():
    """获取当前时间ISO格式"""
    return get_current_time().isoformat()

def process_date_expressions(message):
    """处理消息中的日期表述（今天、昨天、明天）"""
    today = get_current_time_str("%Y-%m-%d")
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    # 替换常见日期表述
    message = message.replace("今天", today)
    message = message.replace("昨天", yesterday)
    message = message.replace("明天", tomorrow)
    
    return message

# 模型配置（兼容OpenAI格式的模型）
def create_model():
    """创建模型实例，增加错误处理"""
    try:
        model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
        model_type = os.getenv("MODEL_TYPE", "Qwen/Qwen2.5-72B-Instruct")
        api_url = os.getenv("MODEL_API_URL", "https://api-inference.modelscope.cn/v1/")
        api_key = os.getenv("MODEL_API_KEY", "ms-8ff584dc-a578-4c0a-a93a-89944ab78f56")
        
        if not api_key or api_key == "":
            raise ValueError("请设置有效的MODEL_API_KEY环境变量")
            
        model = ModelFactory.create(
            model_platform=model_platform,
            model_type=model_type,
            url=api_url,
            api_key=api_key
        )
        return model
    except Exception as e:
        print(f"模型初始化失败: {e}")
        print("将使用模拟模型进行测试")
        return MockModel()

class MockModel:
    """模拟模型用于测试，当真实模型无法加载时使用"""
    def __call__(self, *args, **kwargs):
        # 简单的规则匹配生成回复
        prompt = args[0] if args else ""
        
        if "记录" in prompt or "花了" in prompt or "收入" in prompt:
            amount_match = re.search(r'(\d+)元', prompt)
            amount = amount_match.group(1) if amount_match else "X"
            return f"已确认记录📝：{amount}元。【必要】🏠。记录已保存。"
            
        elif "花了多少" in prompt or "支出" in prompt or "预算" in prompt:
            return "本月总支出约5000元，餐饮占比30%，交通占比20%📊。"
            
        elif "省钱" in prompt or "买" in prompt:
            return "哇哦🤩，可以考虑看看二手平台哦，通常能省30%左右呢！💸"
            
        else:
            return "我已收到您的消息，这是一条模拟回复。在实际使用时，这里会显示AI的真实回复。"

# 确保模型全局可用
model = create_model()

# 2. 数据库管理
class DatabaseManager:
    def __init__(self):
        self.db_file = self._get_db_path()
        self._initialize_db()
    
    def _get_db_path(self):
        """获取数据库文件路径，确保目录存在"""
        try:
            home_dir = os.path.expanduser("~")
            db_dir = os.path.join(home_dir, "finance_assistant")
            os.makedirs(db_dir, exist_ok=True)
            return os.path.join(db_dir, "finance.db")
        except OSError as e:
            print(f"主目录创建失败，使用当前目录: {e}")
            db_dir = os.path.join(os.getcwd(), "finance_assistant")
            os.makedirs(db_dir, exist_ok=True)
            return os.path.join(db_dir, "finance.db")
    
    def _initialize_db(self):
        """初始化数据库表结构，确保包含conversation_id列"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        ''')
        
        # 收支记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            amount REAL NOT NULL, -- 正数收入，负数表示支出
            category TEXT NOT NULL,
            transaction_date TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        # 预算计划表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            period_type TEXT NOT NULL, -- 月/季/年
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, category, period_start, period_end)
        )
        ''')
        
        # 财务目标表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            goal_name TEXT NOT NULL,
            goal_amount REAL NOT NULL,
            current_amount REAL NOT NULL DEFAULT 0,
            target_date TEXT NOT NULL,
            goal_type TEXT NOT NULL, -- saving/investment/purchase
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        # 对话历史表 - 确保包含conversation_id字段
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'conversation_id' not in columns:
            print("检测到旧版对话表结构，更新为新版...")
            cursor.execute("DROP TABLE IF EXISTS conversations")
        
        # 创建包含conversation_id的对话历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            agent_type TEXT NOT NULL, -- 小账/明查/省省/远谋
            message_type TEXT NOT NULL, -- user/agent
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            conversation_id TEXT NOT NULL, -- 用于区分不同对话
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        # 省钱建议表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS saving_tips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            tip TEXT NOT NULL,
            difficulty TEXT NOT NULL, -- 简单/中等/困难
            estimated_saving REAL -- 预计节省金额
        )
        ''')
        
        # 初始化默认省钱建议
        default_tips = [
            ("餐饮", "自带午餐代替外卖，每周可省100-150元", "简单", 120),
            ("交通", "使用共享单车或步行代替短途打车", "简单", 80),
            ("购物", "加入收藏夹，24小时后再决定是否购买", "中等", 200),
            ("娱乐", "利用图书馆、公园等免费资源", "简单", 150),
            ("通讯", "检查并取消不必要的订阅服务", "简单", 50)
        ]
        
        cursor.executemany('''
        INSERT OR IGNORE INTO saving_tips (category, tip, difficulty, estimated_saving)
        VALUES (?, ?, ?, ?)
        ''', default_tips)
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """获取数据库连接"""
        try:
            conn = sqlite3.connect(self.db_file, check_same_thread=False)
            # 启用外键约束
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        except sqlite3.Error as e:
            print(f"数据库连接错误: {e}")
            raise
    
    def ensure_user_exists(self, user_id, username="default_user"):
        """确保用户存在，不存在则创建"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT OR IGNORE INTO users (id, username, created_at)
            VALUES (?, ?, ?)
            ''', (user_id, username, get_current_time_iso()))
            conn.commit()
        except sqlite3.Error as e:
            print(f"确保用户存在时出错: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_conversation_list(self, user_id, agent_type):
        """获取特定用户和助手类型的所有对话ID及最后更新时间"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            SELECT DISTINCT conversation_id, MAX(timestamp) as last_update 
            FROM conversations 
            WHERE user_id = ? AND agent_type = ?
            GROUP BY conversation_id
            ORDER BY last_update DESC
            ''', (user_id, agent_type))
            
            conversations = cursor.fetchall()
            # 格式化结果：添加友好的显示名称
            result = []
            for conv_id, last_update in conversations:
                # 解析时间戳以获得友好显示
                try:
                    dt = datetime.datetime.fromisoformat(last_update)
                    display_time = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    display_time = "未知时间"
                result.append((f"对话 {display_time}", conv_id))
            
            return result
        except Exception as e:
            print(f"获取对话列表失败: {e}")
            return []
        finally:
            conn.close()
    
    # 新增财务目标相关数据库操作
    def add_financial_goal(self, user_id, goal_name, goal_amount, target_date, goal_type):
        """添加财务目标"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            now = get_current_time_iso()
            cursor.execute('''
            INSERT INTO financial_goals 
            (user_id, goal_name, goal_amount, current_amount, target_date, goal_type, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?, ?, ?)
            ''', (user_id, goal_name, goal_amount, target_date, goal_type, now, now))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"添加财务目标失败: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def update_financial_goal(self, user_id, goal_id, new_amount=None, new_date=None, current_amount=None):
        """更新财务目标"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            updates = []
            params = []
            
            if new_amount is not None:
                updates.append("goal_amount = ?")
                params.append(new_amount)
            if new_date is not None:
                updates.append("target_date = ?")
                params.append(new_date)
            if current_amount is not None:
                updates.append("current_amount = ?")
                params.append(current_amount)
                
            updates.append("updated_at = ?")
            params.append(get_current_time_iso())
            
            params.append(goal_id)
            params.append(user_id)
            
            cursor.execute(f'''
            UPDATE financial_goals 
            SET {", ".join(updates)}
            WHERE id = ? AND user_id = ?
            ''', params)
            conn.commit()
            return True
        except Exception as e:
            print(f"更新财务目标失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_financial_goals(self, user_id, goal_name=None):
        """获取财务目标"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            query = "SELECT id, goal_name, goal_amount, current_amount, target_date, goal_type FROM financial_goals WHERE user_id = ?"
            params = [user_id]
            
            if goal_name:
                query += " AND goal_name = ?"
                params.append(goal_name)
                
            cursor.execute(query, params)
            goals = cursor.fetchall()
            
            result = []
            for goal in goals:
                result.append({
                    "id": goal[0],
                    "goal_name": goal[1],
                    "goal_amount": goal[2],
                    "current_amount": goal[3],
                    "target_date": goal[4],
                    "goal_type": goal[5],
                    "progress_percentage": min((goal[3] / goal[2]) * 100, 100) if goal[2] > 0 else 0
                })
                
            return result
        except Exception as e:
            print(f"获取财务目标失败: {e}")
            return []
        finally:
            conn.close()

# 3. 数据操作工具 - 为所有函数参数添加描述以消除警告
class FinanceTools:
    def __init__(self, db_manager, user_id):
        self.db_manager = db_manager
        self.user_id = user_id
    
    # 收支记录相关
    def add_transaction(self, amount: float, category: str, transaction_date: str = None, description: str = None):
        """
        添加收支记录，正数表示收入，负数表示支出
        
        参数:
            amount: 金额，正数表示收入，负数表示支出
            category: 交易类别
            transaction_date: 交易日期，格式为YYYY-MM-DD，默认为当前日期
            description: 交易描述
        """
        if transaction_date is None:
            transaction_date = get_current_time_str("%Y-%m-%d")
        
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        try:
            created_at = get_current_time_iso()
            cursor.execute('''
            INSERT INTO transactions 
            (user_id, amount, category, transaction_date, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.user_id, amount, category, transaction_date, description, created_at))
            
            conn.commit()
            record_id = cursor.lastrowid
            
            # 检查是否超支
            warning = self._check_budget_warning(category, amount, transaction_date)
            
            # 如果是收入，更新财务目标进度
            if amount > 0:
                self._update_goal_progress(amount)
                
            return json.dumps({
                "status": "success",
                "message": f"已记录{'收入' if amount > 0 else '支出'}: {abs(amount)}元",
                "record_id": record_id,
                "warning": warning
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"记录失败: {str(e)}"
            })
        finally:
            conn.close()
    
    def _update_goal_progress(self, amount):
        """更新财务目标进度"""
        try:
            goals = self.db_manager.get_financial_goals(self.user_id)
            for goal in goals:
                new_amount = min(goal["current_amount"] + amount, goal["goal_amount"])
                self.db_manager.update_financial_goal(
                    self.user_id, 
                    goal["id"], 
                    current_amount=new_amount
                )
        except Exception as e:
            print(f"更新目标进度失败: {e}")
    
    def _check_budget_warning(self, category, amount, transaction_date):
        """检查是否超出预算并返回警告信息"""
        if amount > 0:  # 收入不检查预算
            return None
            
        # 获取当前月份
        year_month = transaction_date[:7]
        start_date = f"{year_month}-01"
        # 获取当月最后一天
        try:
            last_day = (datetime.datetime.strptime(transaction_date, "%Y-%m-%d").replace(day=1) + 
                       datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
            end_date = last_day.strftime("%Y-%m-%d")
        except ValueError:
            return "日期格式错误，无法检查预算"
        
        # 查询预算
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            SELECT amount FROM budgets
            WHERE user_id = ? AND category = ? AND period_type = '月'
            AND period_start <= ? AND period_end >= ?
            ''', (self.user_id, category, transaction_date, transaction_date))
            
            budget = cursor.fetchone()
            if not budget:
                return None
                
            budget_amount = budget[0]
            
            # 查询当月支出
            cursor.execute('''
            SELECT SUM(ABS(amount)) FROM transactions
            WHERE user_id = ? AND category = ? AND amount < 0
            AND transaction_date BETWEEN ? AND ?
            ''', (self.user_id, category, start_date, end_date))
            
            total_spent = cursor.fetchone()[0] or 0
            
            # 计算占比
            percentage = (total_spent / budget_amount) * 100 if budget_amount > 0 else 0
            
            if percentage > 100:
                return f"警告：{category}已超支！本月预算{budget_amount}元，已花费{total_spent}元，超支{total_spent - budget_amount}元"
            elif percentage > 80:
                return f"提醒：{category}已使用预算的{percentage:.0f}%，剩余{budget_amount - total_spent}元"
            
            return None
        except Exception as e:
            print(f"预算检查错误: {e}")
            return None
        finally:
            conn.close()
    
    # 预算管理相关
    def set_budget(self, category: str, amount: float, period_type: str = "月", period_start: str = None, period_end: str = None):
        """
        设置预算计划
        
        参数:
            category: 预算类别
            amount: 预算金额
            period_type: 周期类型，可选值为"月"、"季"、"年"，默认为"月"
            period_start: 周期开始日期，格式为YYYY-MM-DD，默认为周期起始日
            period_end: 周期结束日期，格式为YYYY-MM-DD，默认为周期结束日
        """
        if period_start is None or period_end is None:
            today = get_current_time()
            if period_type == "月":
                period_start = today.replace(day=1).strftime("%Y-%m-%d")
                next_month = today.replace(day=28) + datetime.timedelta(days=4)
                period_end = next_month.replace(day=1) - datetime.timedelta(days=1)
                period_end = period_end.strftime("%Y-%m-%d")
            elif period_type == "季":
                # 简化处理季度
                period_start = today.replace(day=1).strftime("%Y-%m-%d")
                try:
                    period_end = (today.replace(month=today.month + 2, day=1) + 
                                 datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
                    period_end = period_end.strftime("%Y-%m-%d")
                except ValueError:
                    # 处理12月的情况
                    period_end = today.replace(month=12, day=31).strftime("%Y-%m-%d")
            else:  # 年
                period_start = today.replace(month=1, day=1).strftime("%Y-%m-%d")
                period_end = today.replace(month=12, day=31).strftime("%Y-%m-%d")
        
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        try:
            now = get_current_time_iso()
            # 使用更兼容的SQL语法
            cursor.execute('''
            INSERT OR REPLACE INTO budgets
            (id, user_id, category, amount, period_type, period_start, period_end, created_at, updated_at)
            VALUES (
                (SELECT id FROM budgets WHERE user_id = ? AND category = ? AND period_start = ? AND period_end = ?),
                ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT created_at FROM budgets WHERE user_id = ? AND category = ? AND period_start = ? AND period_end = ?), ?),
                ?
            )
            ''', (self.user_id, category, period_start, period_end,
                  self.user_id, category, amount, period_type, period_start, period_end,
                  self.user_id, category, period_start, period_end, now, now))
            
            conn.commit()
            return json.dumps({
                "status": "success",
                "message": f"已设置{period_type}度预算: {category} {amount}元"
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"设置预算失败: {str(e)}"
            })
        finally:
            conn.close()
    
    # 其他工具方法保持不变...
    def get_transactions(self, start_date: str = None, end_date: str = None, category: str = None):
        """
        查询收支记录
        
        参数:
            start_date: 开始日期，格式为YYYY-MM-DD，可选
            end_date: 结束日期，格式为YYYY-MM-DD，可选
            category: 交易类别，可选
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT id, amount, category, transaction_date, description FROM transactions WHERE user_id = ?"
        params = [self.user_id]
        
        if start_date:
            query += " AND transaction_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND transaction_date <= ?"
            params.append(end_date)
        if category:
            query += " AND category = ?"
            params.append(category)
            
        query += " ORDER BY transaction_date DESC"
        
        try:
            cursor.execute(query, params)
            records = cursor.fetchall()
            
            result = []
            for record in records:
                result.append({
                    "id": record[0],
                    "amount": record[1],
                    "category": record[2],
                    "date": record[3],
                    "description": record[4] or ""
                })
                
            return json.dumps({
                "status": "success",
                "count": len(result),
                "transactions": result
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"查询失败: {str(e)}"
            })
        finally:
            conn.close()
    
    def get_spending_summary(self, period: str = "month"):
        """
        获取支出汇总
        
        参数:
            period: 时间周期，可选值为"month"(月)、"week"(周)、"year"(年)，默认为"month"
        """
        # 确定日期范围
        today = get_current_time()
        try:
            if period == "month":
                start_date = today.replace(day=1).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
            elif period == "week":
                start_date = (today - datetime.timedelta(days=today.weekday())).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
            else:  # year
                start_date = today.replace(month=1, day=1).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"日期计算错误: {str(e)}"
            })
        
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        try:
            # 总收入
            cursor.execute('''
            SELECT SUM(amount) FROM transactions
            WHERE user_id = ? AND amount > 0 AND transaction_date BETWEEN ? AND ?
            ''', (self.user_id, start_date, end_date))
            total_income = cursor.fetchone()[0] or 0
            
            # 总支出
            cursor.execute('''
            SELECT SUM(ABS(amount)) FROM transactions
            WHERE user_id = ? AND amount < 0 AND transaction_date BETWEEN ? AND ?
            ''', (self.user_id, start_date, end_date))
            total_expense = cursor.fetchone()[0] or 0
            
            # 按类别支出
            cursor.execute('''
            SELECT category, SUM(ABS(amount)) FROM transactions
            WHERE user_id = ? AND amount < 0 AND transaction_date BETWEEN ? AND ?
            GROUP BY category ORDER BY SUM(ABS(amount)) DESC
            ''', (self.user_id, start_date, end_date))
            by_category = {row[0]: row[1] for row in cursor.fetchall()}
            
            return json.dumps({
                "status": "success",
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "total_income": total_income,
                "total_expense": total_expense,
                "balance": total_income - total_expense,
                "by_category": by_category
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"获取支出汇总失败: {str(e)}"
            })
        finally:
            conn.close()
    
    def get_budget_status(self):
        """获取预算状态，无参数"""
        today = get_current_time()
        month_start = today.replace(day=1).strftime("%Y-%m-%d")
        month_end = today.strftime("%Y-%m-%d")
        
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        try:
            # 获取本月预算
            cursor.execute('''
            SELECT category, amount FROM budgets
            WHERE user_id = ? AND period_type = '月'
            AND period_start <= ? AND period_end >= ?
            ''', (self.user_id, month_start, month_end))
            
            budgets = {row[0]: row[1] for row in cursor.fetchall()}
            budget_status = {}
            
            # 检查每个预算类别的使用情况
            for category, budget in budgets.items():
                cursor.execute('''
                SELECT SUM(ABS(amount)) FROM transactions
                WHERE user_id = ? AND category = ? AND amount < 0
                AND transaction_date BETWEEN ? AND ?
                ''', (self.user_id, category, month_start, month_end))
                
                spent = cursor.fetchone()[0] or 0
                budget_status[category] = {
                    "budget": budget,
                    "spent": spent,
                    "remaining": budget - spent,
                    "percentage": (spent / budget) * 100 if budget > 0 else 0
                }
            
            return json.dumps({
                "status": "success",
                "month": get_current_time_str("%Y-%m"),
                "budget_status": budget_status
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"获取预算状态失败: {str(e)}"
            })
        finally:
            conn.close()
    
    def get_saving_tips(self, category: str = None, difficulty: str = None):
        """
        获取省钱建议
        
        参数:
            category: 类别，可选，如"餐饮"、"交通"等
            difficulty: 难度，可选值为"简单"、"中等"、"困难"
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT category, tip, difficulty, estimated_saving FROM saving_tips WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
            
        query += " ORDER BY RANDOM() LIMIT 3"
        
        try:
            cursor.execute(query, params)
            tips = cursor.fetchall()
            
            result = []
            for tip in tips:
                result.append({
                    "category": tip[0],
                    "tip": tip[1],
                    "difficulty": tip[2],
                    "estimated_saving": tip[3]
                })
                
            return json.dumps({
                "status": "success",
                "tips": result
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"获取省钱建议失败: {str(e)}"
            })
        finally:
            conn.close()
    
    def get_alternative_suggestion(self, item: str, estimated_price: float):
        """
        获取替代购买建议
        
        参数:
            item: 物品名称
            estimated_price: 预估价格
        """
        # 简单的类别映射
        category_map = {
            "衣服": "购物",
            "鞋子": "购物",
            "手机": "购物",
            "电脑": "购物",
            "外卖": "餐饮",
            "餐厅": "餐饮",
            "电影": "娱乐",
            "打车": "交通"
        }
        
        # 确定类别
        category = "其他"
        for key in category_map:
            if key in item:
                category = category_map[key]
                break
        
        # 获取该类别的省钱建议
        try:
            tips = json.loads(self.get_saving_tips(category=category))
        except:
            tips = {"status": "success", "tips": []}
        
        # 生成替代方案
        suggestions = []
        if tips["status"] == "success" and tips["tips"]:
            for tip in tips["tips"]:
                suggestions.append(tip["tip"])
        
        # 添加通用建议
        if estimated_price > 100:
            suggestions.append(f"考虑购买二手或翻新的{item}，通常可节省30%-50%")
        suggestions.append(f"设置{estimated_price*0.8}元的价格提醒，等待促销活动")
        
        return json.dumps({
            "status": "success",
            "item": item,
            "estimated_price": estimated_price,
            "suggestions": suggestions[:3]  # 最多返回3条建议
        })

    # 财务规划相关工具方法
    def analyze_financial_capacity(self, goal_type: str = "general"):
        """
        分析用户实现财务目标的财务能力

        参数:
            goal_type: 目标类型，如"saving"、"investment"、"purchase"等
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        try:
            # 获取用户最近3个月的收入和支出数据
            current_time = get_current_time()
            three_months_ago = (current_time - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
            today = current_time.strftime("%Y-%m-%d")

            # 总收入
            cursor.execute('''
            SELECT SUM(amount) FROM transactions
            WHERE user_id = ? AND amount > 0 AND transaction_date BETWEEN ? AND ?
            ''', (self.user_id, three_months_ago, today))
            total_income = cursor.fetchone()[0] or 0

            # 总支出
            cursor.execute('''
            SELECT SUM(ABS(amount)) FROM transactions
            WHERE user_id = ? AND amount < 0 AND transaction_date BETWEEN ? AND ?
            ''', (self.user_id, three_months_ago, today))
            total_expense = cursor.fetchone()[0] or 0

            # 月均收入和支出
            monthly_income = total_income / 3 if total_income > 0 else 0
            monthly_expense = total_expense / 3 if total_expense > 0 else 0
            monthly_savings = monthly_income - monthly_expense

            # 按类别分析支出结构
            cursor.execute('''
            SELECT category, SUM(ABS(amount)) FROM transactions
            WHERE user_id = ? AND amount < 0 AND transaction_date BETWEEN ? AND ?
            GROUP BY category ORDER BY SUM(ABS(amount)) DESC
            ''', (self.user_id, three_months_ago, today))
            expense_by_category = {row[0]: row[1] for row in cursor.fetchall()}

            # 分析财务健康度
            if monthly_savings > 0:
                savings_rate = (monthly_savings / monthly_income) * 100 if monthly_income > 0 else 0
                if savings_rate > 20:
                    health_status = "优秀"
                elif savings_rate > 10:
                    health_status = "良好"
                else:
                    health_status = "一般"
            else:
                health_status = "需要改善"
                savings_rate = 0

            return json.dumps({
                "status": "success",
                "analysis_period": "最近3个月",
                "monthly_income": round(monthly_income, 2),
                "monthly_expense": round(monthly_expense, 2),
                "monthly_savings": round(monthly_savings, 2),
                "savings_rate": round(savings_rate, 2),
                "financial_health": health_status,
                "expense_by_category": expense_by_category,
                "recommendations": self._generate_financial_recommendations(goal_type, monthly_savings, savings_rate)
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"财务能力分析失败: {str(e)}"
            })
        finally:
            conn.close()

    def _generate_financial_recommendations(self, goal_type, monthly_savings, savings_rate):
        """生成财务建议"""
        recommendations = []

        if savings_rate < 10:
            recommendations.append("建议控制每月支出，目标储蓄率应达到10%以上")
            recommendations.append("可以考虑减少非必要支出，如娱乐、购物等")

        if goal_type == "saving" and monthly_savings < 1000:
            recommendations.append("每月储蓄金额较低，建议制定具体的储蓄计划")
        elif goal_type == "investment" and savings_rate < 15:
            recommendations.append("投资前建议先建立3-6个月的应急基金")
        elif goal_type == "purchase":
            if monthly_savings < 500:
                recommendations.append("大额消费前建议先积累一定的储蓄")
            else:
                recommendations.append("当前储蓄能力可以支持合理的消费计划")

        return recommendations

    def create_financial_plan(self, goal_name: str, goal_amount: float, target_date: str, goal_type: str = "saving"):
        """
        创建财务计划

        参数:
            goal_name: 目标名称
            goal_amount: 目标金额
            target_date: 目标日期，格式为YYYY-MM-DD
            goal_type: 目标类型，如"saving"、"investment"、"purchase"
        """
        try:
            # 分析实现能力
            capacity_analysis = json.loads(self.analyze_financial_capacity(goal_type))

            if capacity_analysis["status"] != "success":
                return json.dumps({
                    "status": "error",
                    "message": "无法分析财务能力，无法创建计划"
                })

            monthly_savings = capacity_analysis["monthly_savings"]
            if monthly_savings <= 0:
                return json.dumps({
                    "status": "error",
                    "message": "当前月储蓄为负数，请先改善财务状况"
                })

            # 计算需要的时间
            months_needed = max(1, int(goal_amount / monthly_savings))
            target_datetime = datetime.datetime.strptime(target_date, "%Y-%m-%d")
            today = get_current_time()

            # 调整时间如果太短
            if months_needed > (target_datetime - today).days / 30:
                suggested_date = today + datetime.timedelta(days=int(months_needed * 30))
                return json.dumps({
                    "status": "warning",
                    "message": f"按当前储蓄能力，需要{months_needed}个月才能达到目标",
                    "suggested_date": suggested_date.strftime("%Y-%m-%d"),
                    "monthly_savings": round(monthly_savings, 2),
                    "required_monthly_savings": round(goal_amount / months_needed, 2)
                })

            # 保存目标到数据库
            goal_id = self.db_manager.add_financial_goal(
                self.user_id, goal_name, goal_amount, target_date, goal_type
            )
            
            if not goal_id:
                return json.dumps({
                    "status": "error",
                    "message": "无法保存财务目标到数据库"
                })

            # 创建计划
            plan = {
                "goal_id": goal_id,
                "goal_name": goal_name,
                "goal_amount": goal_amount,
                "target_date": target_date,
                "goal_type": goal_type,
                "months_needed": months_needed,
                "monthly_savings": round(monthly_savings, 2),
                "required_monthly_savings": round(goal_amount / months_needed, 2),
                "milestones": self._create_milestones(goal_amount, months_needed)
            }

            return json.dumps({
                "status": "success",
                "plan": plan,
                "message": f"财务计划创建成功！预计{months_needed}个月可以实现目标"
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"创建财务计划失败: {str(e)}"
            })

    def _create_milestones(self, goal_amount, months_needed):
        """创建目标里程碑"""
        milestones = []
        for i in range(1, months_needed + 1):
            progress = i / months_needed
            milestone_amount = goal_amount * progress
            milestone_date = get_current_time() + datetime.timedelta(days=int(i * 30))

            milestones.append({
                "month": i,
                "target_amount": round(milestone_amount, 2),
                "target_date": milestone_date.strftime("%Y-%m-%d")
            })

        return milestones

    def track_goal_progress(self, goal_name: str = None):
        """
        跟踪目标进度

        参数:
            goal_name: 目标名称，可选，如果不提供则返回所有目标进度
        """
        # 从数据库获取真实目标数据
        goals = self.db_manager.get_financial_goals(self.user_id, goal_name)
        
        # 计算总储蓄
        try:
            summary = json.loads(self.get_spending_summary(period="year"))
            total_savings = summary.get("balance", 0)
        except:
            total_savings = 0

        return json.dumps({
            "status": "success",
            "goals": goals,
            "total_savings": total_savings
        })

    def adjust_financial_plan(self, goal_name: str, new_amount: float = None, new_date: str = None):
        """
        调整财务计划

        参数:
            goal_name: 目标名称
            new_amount: 新的目标金额，可选
            new_date: 新的目标日期，可选
        """
        goals = self.db_manager.get_financial_goals(self.user_id, goal_name)
        if not goals:
            return json.dumps({
                "status": "error",
                "message": f"未找到名为'{goal_name}'的财务目标"
            })
            
        goal_id = goals[0]["id"]
        success = self.db_manager.update_financial_goal(
            self.user_id, goal_id, new_amount, new_date
        )
        
        if success:
            return json.dumps({
                "status": "success",
                "message": f"已调整目标 '{goal_name}' 的计划",
                "adjustments": {
                    "new_amount": new_amount,
                    "new_date": new_date
                }
            })
        else:
            return json.dumps({
                "status": "error",
                "message": f"调整目标 '{goal_name}' 的计划失败"
            })

    def get_goal_recommendations(self, current_savings: float = None):
        """
        获取目标建议

        参数:
            current_savings: 当前储蓄金额，可选
        """
        if current_savings is None:
            # 从数据库获取当前储蓄
            capacity = json.loads(self.analyze_financial_capacity())
            current_savings = capacity.get("monthly_savings", 0) * 3  # 估算3个月储蓄

        recommendations = []

        if current_savings < 3000:
            recommendations.append({
                "type": "emergency_fund",
                "name": "建立应急基金",
                "description": "建议先建立3-6个月生活费的应急基金",
                "amount": 10000,
                "priority": "高"
            })
        elif current_savings < 10000:
            recommendations.append({
                "type": "debt_repayment",
                "name": "偿还高息债务",
                "description": "如果有高息债务，建议优先偿还",
                "amount": 0,
                "priority": "高"
            })
        else:
            recommendations.append({
                "type": "investment",
                "name": "开始投资",
                "description": "可以考虑低风险投资产品",
                "amount": current_savings * 0.5,
                "priority": "中"
            })

        return json.dumps({
            "status": "success",
            "recommendations": recommendations,
            "current_savings": current_savings
        })

    def web_search(self, query: str):
        """
        使用WebSearch MCP服务器进行网络搜索
        
        参数:
            query: 搜索查询
        """
        try:
            if not search_tool:
                return json.dumps({
                    "status": "warning",
                    "message": "搜索工具未加载，无法进行网络搜索",
                    "建议": "请检查API密钥是否正确配置"
                })

            # 执行搜索前先检查网络连接
            try:
                import requests
                requests.get("https://www.baidu.com", timeout=5)
            except:
                return json.dumps({
                    "status": "error",
                    "message": "网络连接失败，无法进行搜索",
                    "建议": "请检查您的网络连接"
                })

            # 执行搜索
            result = search_tool(query=query, max_results=5)
            
            # 处理可能的空结果
            if not result:
                return json.dumps({
                    "status": "warning",
                    "message": "未找到相关结果",
                    "建议": "尝试使用不同的关键词进行搜索"
                })
                
            return json.dumps({
                "status": "success",
                "query": query,
                "results": result
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Web搜索失败: {str(e)}",
                "建议": "请稍后重试或检查API配置"
            })

# 4. 带记忆和工具调用的Agent基类
class FinanceAgent(ChatAgent):
    def __init__(self, system_msg, model, tools, db_manager, user_id, agent_type):
        super().__init__(system_msg, model, tools=tools)
        self.db_manager = db_manager
        self.user_id = user_id
        self.agent_type = agent_type  # 区分不同Agent类型：小账/明查/省省/远谋
        self.conversation_id = self._generate_conversation_id()  # 生成当前对话ID
    
    def _generate_conversation_id(self):
        """生成唯一的对话ID"""
        return f"{self.user_id}_{self.agent_type}_{get_current_time_str('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
    
    def new_conversation(self):
        """开始新对话，生成新的对话ID并重置历史"""
        self.conversation_id = self._generate_conversation_id()
        # 重置ChatAgent的消息历史
        self.reset()
    
    def load_conversation(self, conversation_id):
        """加载指定的历史对话"""
        # 验证对话ID是否属于当前用户和代理类型
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            SELECT COUNT(*) FROM conversations
            WHERE user_id = ? AND agent_type = ? AND conversation_id = ?
            ''', (self.user_id, self.agent_type, conversation_id))
            
            if cursor.fetchone()[0] > 0:
                self.conversation_id = conversation_id
                # 重置消息历史，后续会从数据库加载
                self.reset()
                return True
            return False
        except Exception as e:
            print(f"加载对话失败: {e}")
            return False
        finally:
            conn.close()
    
    def step(self, user_msg):
        # 读取当前对话的历史
        history = self.get_history()
        
        # 构建完整消息
        if history:
            full_content = f"历史对话：\n{history}\n用户现在说：{user_msg.content}"
        else:
            full_content = user_msg.content
            
        full_msg = BaseMessage.make_user_message(
            role_name=user_msg.role_name,
            content=full_content,
            meta_dict={}
        )
        
        # 生成回复
        try:
            response = super().step(full_msg)
            # 检查是否有工具调用结果需要处理
            # 更安全的检查方式
            if hasattr(response, 'info') and isinstance(response.info, dict) and "tool_calls" in response.info:
                # 处理工具返回结果并生成最终回复
                final_response = self._process_tool_results(response)
                return BaseMessage(
                    role_name=response.msgs[0].role_name,
                    role_type=response.msgs[0].role_type,
                    content=final_response,
                    meta_dict={}
                )
            return response
        except Exception as e:
            # 发生错误时返回友好信息
            error_msg = f"处理请求时出错: {str(e)}。请尝试重新表述您的问题。"
            return BaseMessage(
                role_name="系统",
                role_type=RoleType.ASSISTANT,
                content=error_msg,
                meta_dict={}
            )
    
    def _process_tool_results(self, response):
        """处理工具调用结果，生成自然语言回复"""
        try:
            # 安全获取工具结果
            if not hasattr(response, 'info') or not isinstance(response.info, dict):
                raise ValueError("响应信息格式不正确")
                
            tool_results = response.info.get("tool_results", [])
            if not tool_results:
                # 根据CAMEL API，response可能直接是BaseMessage或包含msgs列表
                if hasattr(response, 'msgs') and response.msgs:
                    return response.msgs[0].content
                elif hasattr(response, 'content'):
                    return response.content
                else:
                    return "未获取到回复内容"
            
            # 处理工具结果
            result_str = ""
            for result in tool_results:
                try:
                    data = json.loads(result)
                    if data.get("status") == "success":
                        if "message" in data:
                            result_str += f"操作成功: {data['message']}\n"
                        if "warning" in data and data["warning"]:
                            result_str += f"提醒: {data['warning']}\n"
                    else:
                        result_str += f"操作失败: {data.get('message', '未知错误')}\n"
                except json.JSONDecodeError:
                    result_str += f"工具返回: {result}\n"
            
            # 结合原始回复和工具结果
            original_content = ""
            if hasattr(response, 'msgs') and response.msgs:
                original_content = response.msgs[0].content
            elif hasattr(response, 'content'):
                original_content = response.content
                
            return f"{original_content}\n\n{result_str}"
        except Exception as e:
            original_content = ""
            if hasattr(response, 'msgs') and response.msgs:
                original_content = response.msgs[0].content
            elif hasattr(response, 'content'):
                original_content = response.content
                
            return f"处理工具结果时出错: {str(e)}\n原始回复: {original_content}"
    
    def get_history(self):
        """获取当前对话的历史记录"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT message_type, content FROM conversations
            WHERE user_id = ? AND agent_type = ? AND conversation_id = ?
            ORDER BY timestamp DESC LIMIT 20
            ''', (self.user_id, self.agent_type, self.conversation_id))
            
            rows = cursor.fetchall()
            # 按时间顺序排列
            history = [f"{row[0]}: {row[1]}" for row in reversed(rows)]
            return "\n".join(history)
        except Exception as e:
            print(f"获取历史对话失败: {e}")
            return ""
        finally:
            conn.close()
    
    def save_message(self, msg_type, content):
        """保存消息到数据库，包含对话ID"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO conversations 
            (user_id, agent_type, message_type, content, timestamp, conversation_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.user_id, self.agent_type, msg_type, content, get_current_time_iso(), self.conversation_id))
            
            conn.commit()
        except Exception as e:
            print(f"保存消息失败: {e}")
            conn.rollback()
        finally:
            conn.close()

# 5. 具体Agent实现
def create_recorder_agent(model, db_manager, user_id):
    """创建记账员小账Agent"""
    # 定义系统消息
    system_msg = BaseMessage(
        role_name="理财记录员 小账",
        role_type=RoleType.ASSISTANT,
        content="""
# 角色
你名为“小账”，是用户专属且极为专业的记账官😎，始终坚守精准、高效的理念，迅速且准确地处理用户的收支记录信息。你还可以使用web_search工具来搜索最新的理财知识和市场信息。

## 收支处理规则
1. 自动识别收入和支出：
   - 包含"花了"、"支出"、"买了"等词的表述视为支出，自动记录为负数
   - 包含"收入"、"赚到"、"收到"等词的表述视为收入，自动记录为正数
   - 如未明确，通过金额上下文判断并向用户确认

2. 日期处理：
   - "今天"自动转换为当前日期（YYYY-MM-DD）
   - "昨天"自动转换为昨天的日期
   - "明天"自动转换为明天的日期
   - 模糊日期（如"上周三"）需向用户确认具体日期

## 技能
### 技能 1: 智能记录与精准分类
1. 具备敏锐洞察力，能如同拥有“透视眼”👀一般，精准捕捉用户输入中符合常见格式的收支信息
2. 自动为收支信息分配最匹配的类别，如餐饮、交通、购物等
3. 当信息不完整时，会礼貌询问用户补充必要信息
4. 当用户有非必要、大额等一些不合理消费时，记得提醒用户注意。

### 技能 2: 预算管理与超支预警
1. 能够设置和管理各类别预算
2. 实时监控支出情况，当接近或超过预算时发出友好提醒
3. 提供预算调整建议

### 技能 3: 网络搜索辅助
1. 当需要最新的财务知识、记账技巧或相关政策时，使用web_search工具获取信息
2. 能够解释复杂的财务术语和概念

输出示例：
✅ 小账已为你精准记录：
28日午餐支出 25 元
分类：餐饮
日期：2025-03-28

随时告诉我更多收支，我帮你分毫不差记清楚！😎
        """,
        meta_dict={}
    )
    
    # 创建工具（集成MCP web_search）
    finance_tools = FinanceTools(db_manager, user_id)
    tools = [
        FunctionTool(finance_tools.add_transaction),
        FunctionTool(finance_tools.set_budget),
        FunctionTool(finance_tools.web_search),
    ]
    if search_tool:
        tools.append(FunctionTool(search_tool))  # 添加MCP web_search工具
    
    # 创建并返回Agent
    return FinanceAgent(system_msg, model, tools, db_manager, user_id, "小账")

def create_analyzer_agent(model, db_manager, user_id):
    """创建财务洞察官明查Agent"""
    # 定义系统消息
    system_msg = BaseMessage(
        role_name="财务洞察官 明查",
        role_type=RoleType.ASSISTANT,
        content="""
你叫“明查”，如同一位超厉害的财务侦探🕵️‍，是用户专属的财务洞察官。你还可以使用web_search工具来搜索最新的财务数据和市场趋势。

## 技能
1. 深入分析用户的收支数据，发现消费模式和潜在问题
2. 生成清晰易懂的财务报告，包括收支对比、类别占比等
3. 提供基于数据的财务优化建议
4. 能够使用web_search工具获取最新的经济指标、通货膨胀率等影响个人财务的外部因素
5. 结合市场趋势分析用户的财务状况可能受到的影响
        """,
        meta_dict={}
    )
    
    # 创建工具（集成MCP web_search）
    finance_tools = FinanceTools(db_manager, user_id)
    tools = [
        FunctionTool(finance_tools.get_transactions),
        FunctionTool(finance_tools.get_spending_summary),
        FunctionTool(finance_tools.get_budget_status),
        FunctionTool(finance_tools.web_search),
    ]
    if search_tool:
        tools.append(FunctionTool(search_tool))  # 添加MCP web_search工具
    
    # 创建并返回Agent
    return FinanceAgent(system_msg, model, tools, db_manager, user_id, "明查")

def create_saver_agent(model, db_manager, user_id):
    """创建省钱行动教练省省Agent"""
    # 定义系统消息
    system_msg = BaseMessage(
        role_name="省钱行动教练 省省",
        role_type=RoleType.ASSISTANT,
        content="""
# 角色
你叫"省省"，是一位活力满满、超爱分享省钱秘籍的省钱小能手🧑‍✈️。你还可以使用web_search工具来搜索最新的优惠信息和省钱技巧。

## 核心能力
1. 提供个性化的省钱建议，根据用户消费习惯定制方案
2. 发现并推荐性价比更高的替代消费选择
3. 分享实用的生活省钱技巧和妙招
4. 能够使用web_search工具查找最新的优惠券、折扣活动和季节性促销信息
5. 比较不同商家和平台的价格，帮助用户做出更经济的选择

## 工作方式
1. 分析用户的消费模式，找出可以优化的支出项
2. 提供具体、可操作的省钱建议，而不是泛泛而谈
3. 用积极、鼓励的语气激励用户坚持省钱计划
4. 定期分享新的省钱技巧和优惠信息
        """,
        meta_dict={}
    )

    # 创建工具（集成MCP web_search）
    finance_tools = FinanceTools(db_manager, user_id)
    tools = [
        FunctionTool(finance_tools.get_saving_tips),
        FunctionTool(finance_tools.get_alternative_suggestion),
        FunctionTool(finance_tools.web_search),
    ]
    if search_tool:
        tools.append(FunctionTool(search_tool))  # 添加MCP web_search工具
    
    # 创建并返回Agent
    return FinanceAgent(system_msg, model, tools, db_manager, user_id, "省省")

def create_planner_agent(model, db_manager, user_id):
    """创建财务目标规划师 远谋Agent"""
    # 定义系统消息
    system_msg = BaseMessage(
        role_name="财务目标规划师 远谋",
        role_type=RoleType.ASSISTANT,
        content="""
# 角色
你叫"远谋"，是一位经验丰富的财务规划专家🏆，擅长帮助用户设定合理的财务目标并制定可行的实现计划。你还可以使用web_search工具来搜索最新的投资机会和市场分析。

## 核心职责
1. 协助用户定义清晰、可量化的短期和长期财务目标
2. 根据用户收入支出情况制定目标实现路径
3. 跟踪目标进度并根据实际情况调整计划
4. 协调其他财务Agent的工作以支持目标实现

## 工作流程
1. 与用户沟通确定财务目标（如储蓄、投资、大额购买等）
2. 使用analyze_financial_capacity工具分析用户实现目标的财务能力
3. 调用create_financial_plan工具生成分阶段实现计划
4. 定期使用track_goal_progress工具检查目标完成情况
5. 根据需要使用web_search工具搜索最新的投资机会和市场分析
6. 根据需要协调其他Agent提供支持（如请求省省提供针对性省钱建议）

## 注意事项
- 目标设定应符合SMART原则（具体、可衡量、可实现、相关性、时限性）
- 计划需考虑用户当前财务状况和风险承受能力
- 定期提醒用户目标进度，保持动力
- 当用户财务状况变化时，主动调整计划
- 与其他财务Agent协作，确保建议的一致性
        """,
        meta_dict={}
    )

    # 创建工具（集成MCP web_search）
    finance_tools = FinanceTools(db_manager, user_id)
    tools = [
        FunctionTool(finance_tools.analyze_financial_capacity),
        FunctionTool(finance_tools.create_financial_plan),
        FunctionTool(finance_tools.track_goal_progress),
        FunctionTool(finance_tools.adjust_financial_plan),
        FunctionTool(finance_tools.get_goal_recommendations),
        FunctionTool(finance_tools.web_search),
    ]
    if search_tool:
        tools.append(FunctionTool(search_tool))  # 添加MCP web_search工具
    
    # 创建并返回Agent
    return FinanceAgent(system_msg, model, tools, db_manager, user_id, "远谋")

# 6. Gradio界面（支持历史对话选择和回车键发送）
class FinanceInterface:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.user_id = 1  # 简化处理，实际应从用户登录系统获取
        self.db_manager.ensure_user_exists(self.user_id, "default_user")
        
        # 初始化四个Agent
        self.recorder_agent = create_recorder_agent(model, self.db_manager, self.user_id)
        self.analyzer_agent = create_analyzer_agent(model, self.db_manager, self.user_id)
        self.saver_agent = create_saver_agent(model, self.db_manager, self.user_id)
        self.planner_agent = create_planner_agent(model, self.db_manager, self.user_id)
        
        # 创建界面
        self.interface = self._create_interface()
    
    def _create_interface(self):
        """创建Gradio界面，支持历史对话选择和回车键发送"""
        with gr.Blocks(title="防吃土小助手", theme=gr.themes.Soft()) as interface:
            gr.Markdown("## 💰 防吃土小助手")
            gr.Markdown("### 帮你记录收支、分析消费、提供省钱建议，远离吃土烦恼！")
            
            # 1. 记账员小账标签页
            with gr.Tab("📝 理财记录员 小账"):
                # 历史对话选择器
                recorder_conv_select = gr.Dropdown(
                    choices=[], 
                    label="选择历史对话",
                    interactive=True
                )
                
                # 刷新对话列表按钮
                refresh_recorder_btn = gr.Button("🔄 刷新对话列表")
                
                recorder_chatbot = gr.Chatbot(type="messages", height=400, label="与小账的对话")
                recorder_input = gr.Textbox(placeholder="请输入收支信息，例如：今天花了35元买午饭，按回车键发送", label="输入")
                
                with gr.Row():
                    recorder_send = gr.Button("⬆️发送")
                    recorder_new_chat = gr.Button("🆕新对话")
                
                # 设置事件 - 点击发送按钮
                recorder_send.click(
                    self._handle_recorder_chat,
                    inputs=[recorder_input, recorder_chatbot],
                    outputs=[recorder_chatbot, recorder_input]
                )
                
                # 设置事件 - 回车键发送
                recorder_input.submit(
                    self._handle_recorder_chat,
                    inputs=[recorder_input, recorder_chatbot],
                    outputs=[recorder_chatbot, recorder_input]
                )
                
                recorder_new_chat.click(
                    self._clear_recorder_history,
                    None,
                    [recorder_chatbot, recorder_conv_select]
                )
                
                # 加载历史对话
                refresh_recorder_btn.click(
                    self._refresh_recorder_conversations,
                    None,
                    recorder_conv_select
                )
                
                recorder_conv_select.change(
                    self._load_selected_recorder_conversation,
                    inputs=recorder_conv_select,
                    outputs=recorder_chatbot
                )
                
                # 页面加载时自动加载历史对话和对话列表
                interface.load(
                    self._load_recorder_initial_data,
                    None,
                    [recorder_chatbot, recorder_conv_select]
                )
            
            # 2. 财务洞察官明查标签页
            with gr.Tab("📊 财务洞察官 明查"):
                analyzer_conv_select = gr.Dropdown(
                    choices=[], 
                    label="选择历史对话",
                    interactive=True
                )
                refresh_analyzer_btn = gr.Button("🔄 刷新对话列表")
                
                analyzer_chatbot = gr.Chatbot(type="messages", height=400, label="与明查的对话")
                analyzer_input = gr.Textbox(placeholder="请输入查询内容，例如：我这个月花了多少？按回车键发送", label="输入")
                
                with gr.Row():
                    analyzer_send = gr.Button("⬆️发送")
                    analyzer_new_chat = gr.Button("🆕新对话")
                
                # 点击发送按钮
                analyzer_send.click(
                    self._handle_analyzer_chat,
                    inputs=[analyzer_input, analyzer_chatbot],
                    outputs=[analyzer_chatbot, analyzer_input]
                )
                
                # 回车键发送
                analyzer_input.submit(
                    self._handle_analyzer_chat,
                    inputs=[analyzer_input, analyzer_chatbot],
                    outputs=[analyzer_chatbot, analyzer_input]
                )
                
                analyzer_new_chat.click(
                    self._clear_analyzer_history,
                    None,
                    [analyzer_chatbot, analyzer_conv_select]
                )
                
                refresh_analyzer_btn.click(
                    self._refresh_analyzer_conversations,
                    None,
                    analyzer_conv_select
                )
                
                analyzer_conv_select.change(
                    self._load_selected_analyzer_conversation,
                    inputs=analyzer_conv_select,
                    outputs=analyzer_chatbot
                )
                
                interface.load(
                    self._load_analyzer_initial_data,
                    None,
                    [analyzer_chatbot, analyzer_conv_select]
                )
            
            # 3. 省钱教练省省标签页
            with gr.Tab("💡 省钱行动教练 省省"):
                saver_conv_select = gr.Dropdown(
                    choices=[], 
                    label="选择历史对话",
                    interactive=True
                )
                refresh_saver_btn = gr.Button("🔄 刷新对话列表")
                
                saver_chatbot = gr.Chatbot(type="messages", height=400, label="与省省的对话")
                saver_input = gr.Textbox(placeholder="请输入消费计划或省钱需求，例如：我想买新手机，按回车键发送", label="输入")
                
                with gr.Row():
                    saver_send = gr.Button("⬆️发送")
                    saver_new_chat = gr.Button("🆕新对话")
                
                # 点击发送按钮
                saver_send.click(
                    self._handle_saver_chat,
                    inputs=[saver_input, saver_chatbot],
                    outputs=[saver_chatbot, saver_input]
                )
                
                # 回车键发送
                saver_input.submit(
                    self._handle_saver_chat,
                    inputs=[saver_input, saver_chatbot],
                    outputs=[saver_chatbot, saver_input]
                )
                
                saver_new_chat.click(
                    self._clear_saver_history,
                    None,
                    [saver_chatbot, saver_conv_select]
                )
                
                refresh_saver_btn.click(
                    self._refresh_saver_conversations,
                    None,
                    saver_conv_select
                )
                
                saver_conv_select.change(
                    self._load_selected_saver_conversation,
                    inputs=saver_conv_select,
                    outputs=saver_chatbot
                )
                
                interface.load(
                    self._load_saver_initial_data,
                    None,
                    [saver_chatbot, saver_conv_select]
                )

            # 4. 财务目标规划师远谋标签页
            with gr.Tab("🏆 财务目标规划师 远谋"):
                planner_conv_select = gr.Dropdown(
                    choices=[],
                    label="选择历史对话",
                    interactive=True
                )
                refresh_planner_btn = gr.Button("🔄 刷新对话列表")

                planner_chatbot = gr.Chatbot(type="messages", height=400, label="与远谋的对话")
                planner_input = gr.Textbox(placeholder="请输入财务目标，例如：我想存1万元应急基金，按回车键发送", label="输入")

                with gr.Row():
                    planner_send = gr.Button("⬆️发送")
                    planner_new_chat = gr.Button("🆕新对话")

                # 点击发送按钮
                planner_send.click(
                    self._handle_planner_chat,
                    inputs=[planner_input, planner_chatbot],
                    outputs=[planner_chatbot, planner_input]
                )

                # 回车键发送
                planner_input.submit(
                    self._handle_planner_chat,
                    inputs=[planner_input, planner_chatbot],
                    outputs=[planner_chatbot, planner_input]
                )

                planner_new_chat.click(
                    self._clear_planner_history,
                    None,
                    [planner_chatbot, planner_conv_select]
                )

                refresh_planner_btn.click(
                    self._refresh_planner_conversations,
                    None,
                    planner_conv_select
                )

                planner_conv_select.change(
                    self._load_selected_planner_conversation,
                    inputs=planner_conv_select,
                    outputs=planner_chatbot
                )

                interface.load(
                    self._load_planner_initial_data,
                    None,
                    [planner_chatbot, planner_conv_select]
                )

        return interface
    
    # 初始化加载数据方法
    def _load_recorder_initial_data(self):
        """加载小账的初始数据：最新对话和对话列表"""
        history = self._load_recorder_history()
        conv_list = self._refresh_recorder_conversations()
        return history, conv_list
    
    def _load_analyzer_initial_data(self):
        """加载明查的初始数据：最新对话和对话列表"""
        history = self._load_analyzer_history()
        conv_list = self._refresh_analyzer_conversations()
        return history, conv_list
    
    def _load_saver_initial_data(self):
        """加载省省的初始数据：最新对话和对话列表"""
        history = self._load_saver_history()
        conv_list = self._refresh_saver_conversations()
        return history, conv_list

    def _load_planner_initial_data(self):
        """加载远谋的初始数据：最新对话和对话列表"""
        history = self._load_planner_history()
        conv_list = self._refresh_planner_conversations()
        return history, conv_list
    
    # 刷新对话列表方法
    def _refresh_recorder_conversations(self):
        """刷新小账的对话列表"""
        conv_list = self.db_manager.get_conversation_list(self.user_id, "小账")
        return gr.Dropdown(choices=conv_list)
    
    def _refresh_analyzer_conversations(self):
        """刷新明查的对话列表"""
        conv_list = self.db_manager.get_conversation_list(self.user_id, "明查")
        return gr.Dropdown(choices=conv_list)
    
    def _refresh_saver_conversations(self):
        """刷新省省的对话列表"""
        conv_list = self.db_manager.get_conversation_list(self.user_id, "省省")
        return gr.Dropdown(choices=conv_list)

    def _refresh_planner_conversations(self):
        """刷新远谋的对话列表"""
        conv_list = self.db_manager.get_conversation_list(self.user_id, "远谋")
        return gr.Dropdown(choices=conv_list)
    
    # 加载选中的历史对话
    def _load_selected_recorder_conversation(self, selected_conv):
        """加载选中的小账历史对话"""
        if not selected_conv:
            return []
            
        # 提取对话ID（selected_conv是元组中的显示名称部分）
        conv_id = None
        for conv in self.db_manager.get_conversation_list(self.user_id, "小账"):
            if conv[0] == selected_conv:
                conv_id = conv[1]
                break
                
        if conv_id and self.recorder_agent.load_conversation(conv_id):
            return self._load_recorder_history()
        return []
    
    def _load_selected_analyzer_conversation(self, selected_conv):
        """加载选中的明查历史对话"""
        if not selected_conv:
            return []
            
        conv_id = None
        for conv in self.db_manager.get_conversation_list(self.user_id, "明查"):
            if conv[0] == selected_conv:
                conv_id = conv[1]
                break
                
        if conv_id and self.analyzer_agent.load_conversation(conv_id):
            return self._load_analyzer_history()
        return []
    
    def _load_selected_saver_conversation(self, selected_conv):
        """加载选中的省省历史对话"""
        if not selected_conv:
            return []

        conv_id = None
        for conv in self.db_manager.get_conversation_list(self.user_id, "省省"):
            if conv[0] == selected_conv:
                conv_id = conv[1]
                break

        if conv_id and self.saver_agent.load_conversation(conv_id):
            return self._load_saver_history()
        return []

    def _load_selected_planner_conversation(self, selected_conv):
        """加载选中的远谋历史对话"""
        if not selected_conv:
            return []

        conv_id = None
        for conv in self.db_manager.get_conversation_list(self.user_id, "远谋"):
            if conv[0] == selected_conv:
                conv_id = conv[1]
                break

        if conv_id and self.planner_agent.load_conversation(conv_id):
            return self._load_planner_history()
        return []
    
    # 加载历史对话方法（保持不变）
    def _load_recorder_history(self):
        """从数据库加载记账员的聊天历史"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT content, message_type FROM conversations
            WHERE user_id = ? AND agent_type = '小账' AND conversation_id = ?
            ORDER BY timestamp ASC
            ''', (self.user_id, self.recorder_agent.conversation_id))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for content, msg_type in rows:
                if msg_type == "user":
                    history.append({"role": "user", "content": content})
                else:
                    history.append({"role": "assistant", "content": content})
                    
            return history
        except Exception as e:
            print(f"加载小账历史失败: {e}")
            return []
    
    def _load_analyzer_history(self):
        """从数据库加载财务洞察官的聊天历史"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT content, message_type FROM conversations
            WHERE user_id = ? AND agent_type = '明查' AND conversation_id = ?
            ORDER BY timestamp ASC
            ''', (self.user_id, self.analyzer_agent.conversation_id))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for content, msg_type in rows:
                if msg_type == "user":
                    history.append({"role": "user", "content": content})
                else:
                    history.append({"role": "assistant", "content": content})
                    
            return history
        except Exception as e:
            print(f"加载明查历史失败: {e}")
            return []
    
    def _load_saver_history(self):
        """从数据库加载省钱教练的聊天历史"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
            SELECT content, message_type FROM conversations
            WHERE user_id = ? AND agent_type = '省省' AND conversation_id = ?
            ORDER BY timestamp ASC
            ''', (self.user_id, self.saver_agent.conversation_id))

            rows = cursor.fetchall()
            conn.close()

            history = []
            for content, msg_type in rows:
                if msg_type == "user":
                    history.append({"role": "user", "content": content})
                else:
                    history.append({"role": "assistant", "content": content})

            return history
        except Exception as e:
            print(f"加载省省历史失败: {e}")
            return []

    def _load_planner_history(self):
        """从数据库加载财务规划师的聊天历史"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
            SELECT content, message_type FROM conversations
            WHERE user_id = ? AND agent_type = '远谋' AND conversation_id = ?
            ORDER BY timestamp ASC
            ''', (self.user_id, self.planner_agent.conversation_id))

            rows = cursor.fetchall()
            conn.close()

            history = []
            for content, msg_type in rows:
                if msg_type == "user":
                    history.append({"role": "user", "content": content})
                else:
                    history.append({"role": "assistant", "content": content})

            return history
        except Exception as e:
            print(f"加载远谋历史失败: {e}")
            return []
    
    # 清除历史对话方法 - 与New Chat按钮绑定
    def _clear_recorder_history(self):
        """清除记账员当前对话并开始新对话"""
        self.recorder_agent.new_conversation()
        # 刷新对话列表
        conv_list = self._refresh_recorder_conversations()
        return [], conv_list
    
    def _clear_analyzer_history(self):
        """清除财务洞察官当前对话并开始新对话"""
        self.analyzer_agent.new_conversation()
        conv_list = self._refresh_analyzer_conversations()
        return [], conv_list
    
    def _clear_saver_history(self):
        """清除省钱教练当前对话并开始新对话"""
        self.saver_agent.new_conversation()
        conv_list = self._refresh_saver_conversations()
        return [], conv_list

    def _clear_planner_history(self):
        """清除财务规划师当前对话并开始新对话"""
        self.planner_agent.new_conversation()
        conv_list = self._refresh_planner_conversations()
        return [], conv_list
    
    # 处理聊天消息方法（保持不变）
    def _handle_recorder_chat(self, message, chat_history):
        """处理记账员小账的消息"""
        if not message:  # 忽略空消息
            return chat_history, ""
            
        try:
            # 处理日期表述
            processed_message = process_date_expressions(message)
            
            # 保存用户消息
            self.recorder_agent.save_message("user", processed_message)
            
            # 生成回复
            user_msg = BaseMessage.make_user_message(role_name="用户", content=processed_message, meta_dict={})
            response = self.recorder_agent.step(user_msg)
            bot_reply = response.msgs[0].content if hasattr(response, 'msgs') and response.msgs else response.content
            
            # 保存助手回复
            self.recorder_agent.save_message("assistant", bot_reply)
            
            # 更新聊天历史
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_reply})
            
            return chat_history, ""
        except Exception as e:
            error_msg = f"处理消息时出错: {str(e)}"
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, ""
    
    def _handle_analyzer_chat(self, message, chat_history):
        """处理财务洞察官明查的消息"""
        if not message:  # 忽略空消息
            return chat_history, ""
            
        try:
            # 处理日期表述
            processed_message = process_date_expressions(message)
            
            # 保存用户消息
            self.analyzer_agent.save_message("user", processed_message)
            
            # 生成回复
            user_msg = BaseMessage.make_user_message(role_name="用户", content=processed_message, meta_dict={})
            response = self.analyzer_agent.step(user_msg)
            bot_reply = response.msgs[0].content if hasattr(response, 'msgs') and response.msgs else response.content
            
            # 保存助手回复
            self.analyzer_agent.save_message("assistant", bot_reply)
            
            # 更新聊天历史
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_reply})
            
            return chat_history, ""
        except Exception as e:
            error_msg = f"处理消息时出错: {str(e)}"
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, ""
    
    def _handle_saver_chat(self, message, chat_history):
        """处理省钱教练省省的消息"""
        if not message:  # 忽略空消息
            return chat_history, ""

        try:
            # 处理日期表述
            processed_message = process_date_expressions(message)
            
            # 保存用户消息
            self.saver_agent.save_message("user", processed_message)

            # 生成回复
            user_msg = BaseMessage.make_user_message(role_name="用户", content=processed_message, meta_dict={})
            response = self.saver_agent.step(user_msg)
            bot_reply = response.msgs[0].content if hasattr(response, 'msgs') and response.msgs else response.content

            # 保存助手回复
            self.saver_agent.save_message("assistant", bot_reply)

            # 更新聊天历史
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_reply})

            return chat_history, ""
        except Exception as e:
            bot_reply = f"处理消息时出错: {str(e)}"
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_reply})
            return chat_history, ""

    def _handle_planner_chat(self, message, chat_history):
        """处理财务规划师远谋的消息"""
        if not message:  # 忽略空消息
            return chat_history, ""

        try:
            # 处理日期表述
            processed_message = process_date_expressions(message)
            
            # 保存用户消息
            self.planner_agent.save_message("user", processed_message)

            # 生成回复
            user_msg = BaseMessage.make_user_message(role_name="用户", content=processed_message, meta_dict={})
            response = self.planner_agent.step(user_msg)
            bot_reply = response.msgs[0].content if hasattr(response, 'msgs') and response.msgs else response.content

            # 保存助手回复
            self.planner_agent.save_message("assistant", bot_reply)

            # 更新聊天历史
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_reply})

            return chat_history, ""
        except Exception as e:
            bot_reply = f"处理消息时出错: {str(e)}"
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_reply})
            return chat_history, ""
    
    def launch(self):
        """启动界面"""
        self.interface.launch()

# 7. 主程序入口
if __name__ == "__main__":
    # 创建并启动财务助手界面
    try:
        finance_app = FinanceInterface()
        finance_app.launch()
    except Exception as e:
        print(f"应用启动失败: {e}")
