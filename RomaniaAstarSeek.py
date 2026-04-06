import heapq
from typing import Dict, List, Tuple, Optional, Set
import math

class Node:
    """表示搜索树中的节点"""
    def __init__(self, city: str, parent=None, g_cost: float = 0, h_cost: float = 0):
        self.city = city
        self.parent = parent
        self.g_cost = g_cost  # 从起点到当前节点的实际代价
        self.h_cost = h_cost  # 启发式代价（到目标的估计代价）
        
    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost  # A*算法的评估函数
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost  # 用于优先队列比较

class RomaniaAStar:
    """RomanA*算法"""
    
    def __init__(self):
        # 罗马尼亚城市的启发式值（到Bucharest的直线距离估计）
        self.heuristics = {
            'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
            'Eforie': 161, 'Fagaras': 178, 'Giurgiu': 77, 'Hirsova': 151,
            'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
            'Oradea': 380, 'Pitesti': 98, 'Rimnicu Vilcea': 193, 'Sibiu': 253,
            'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
        }
        
        # 罗马尼亚城市之间的实际距离
        self.graph = {
            'Arad': [('Zerind', 75), ('Sibiu', 140), ('Timisoara', 118)],
            'Zerind': [('Arad', 75), ('Oradea', 71)],
            'Oradea': [('Zerind', 71), ('Sibiu', 151)],
            'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)],
            'Timisoara': [('Arad', 118), ('Lugoj', 111)],
            'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
            'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
            'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
            'Craiova': [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)],
            'Rimnicu Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)],
            'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
            'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)],
            'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90), ('Urziceni', 85)],
            'Giurgiu': [('Bucharest', 90)],
            'Urziceni': [('Bucharest', 85), ('Hirsova', 98), ('Vaslui', 142)],
            'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
            'Eforie': [('Hirsova', 86)],
            'Vaslui': [('Urziceni', 142), ('Iasi', 92)],
            'Iasi': [('Vaslui', 92), ('Neamt', 87)],
            'Neamt': [('Iasi', 87)]
        }
    
    def a_star_search(self, start: str, goal: str) -> Tuple[Optional[List[str]], float]:
        """
        执行A*搜索算法
        
        参数:
            start: 起始城市
            goal: 目标城市
            
        返回:
            (路径, 总代价) 或 (None, 无穷大) 如果未找到路径
        """
        if start not in self.heuristics or goal not in self.heuristics:
            return None, float('inf')
        
        # OPEN列表（优先队列），存储(f_cost, 唯一标识, 节点)
        open_list = []
        # CLOSE列表（已探索节点），用集合存储已访问城市
        close_set = set()
        
        # 创建起始节点
        start_node = Node(start, None, 0, self.heuristics[start])
        heapq.heappush(open_list, (start_node.f_cost, id(start_node), start_node))
        
        # 记录节点与其对应对象的映射，用于快速查找
        node_dict = {start: start_node}
        
        while open_list:
            # 从OPEN列表取出f值最小的节点
            _, _, current_node = heapq.heappop(open_list)
            
            # 如果已经在CLOSE列表中，跳过（避免重复处理）
            if current_node.city in close_set:
                continue
                
            # 将当前节点加入CLOSE列表
            close_set.add(current_node.city)
            
            # 检查是否达到目标
            if current_node.city == goal:
                # 回溯路径
                path = []
                total_cost = current_node.g_cost
                while current_node:
                    path.append(current_node.city)
                    current_node = current_node.parent
                return list(reversed(path)), total_cost
            
            # 扩展当前节点的邻居
            for neighbor, cost in self.graph.get(current_node.city, []):
                if neighbor in close_set:
                    continue
                    
                # 计算新的g值（从起点到邻居的实际代价）
                new_g = current_node.g_cost + cost
                
                # 如果邻居已经在OPEN列表中
                if neighbor in node_dict:
                    existing_node = node_dict[neighbor]
                    if new_g < existing_node.g_cost:
                        # 找到更优路径，更新节点
                        existing_node.parent = current_node
                        existing_node.g_cost = new_g
                        # 重新添加到OPEN列表（更新优先级）
                        heapq.heappush(open_list, (existing_node.f_cost, id(existing_node), existing_node))
                else:
                    # 创建新节点
                    new_node = Node(
                        city=neighbor,
                        parent=current_node,
                        g_cost=new_g,
                        h_cost=self.heuristics[neighbor]
                    )
                    node_dict[neighbor] = new_node
                    heapq.heappush(open_list, (new_node.f_cost, id(new_node), new_node))
        
        return None, float('inf')

def main():
    """主函数"""
    solver = RomaniaAStar()
    
    # 执行A*搜索
    path, total_cost = solver.a_star_search('Arad', 'Bucharest')
    
    if path:
        print("从 Arad 到 Bucharest 的路径:")
        print(" -> ".join(path))
        print(f"总路程: {total_cost} km") 
    else:
        print("未找到路径！")

if __name__ == "__main__":
    main()