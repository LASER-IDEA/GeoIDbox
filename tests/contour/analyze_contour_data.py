"""分析等高线数据结构"""
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Polygon, MultiPoint
import os

def extract_polygons_from_file(filepath):
    """从文件中提取所有polygon"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    polygons = []
    i = 0
    content_len = len(content)

    while i < content_len:
        poly_start = content.find('POLYGON', i)
        if poly_start == -1:
            break

        # 找到匹配的闭合括号
        depth = 0
        in_polygon = False
        end_pos = poly_start

        for j in range(poly_start, content_len):
            char = content[j]
            if char == '(':
                depth += 1
                in_polygon = True
            elif char == ')':
                depth -= 1
                if in_polygon and depth == 0:
                    end_pos = j + 1
                    break

        if end_pos > poly_start:
            wkt_str = content[poly_start:end_pos].strip().rstrip(';,').strip()
            try:
                poly = wkt_loads(wkt_str)
                if isinstance(poly, Polygon) and not poly.is_empty:
                    polygons.append(poly)
            except:
                pass

        i = end_pos if end_pos > poly_start else poly_start + 7

    return polygons

# 分析一个示例文件
if __name__ == "__main__":
    filepath = 'data/00.txt'
    if os.path.exists(filepath):
        polygons = extract_polygons_from_file(filepath)
        print(f"文件 {filepath} 包含 {len(polygons)} 个多边形")

        # 按面积排序
        polygons_sorted = sorted(polygons, key=lambda p: p.area, reverse=True)

        print("\n多边形信息（按面积排序）：")
        for i, poly in enumerate(polygons_sorted[:10]):  # 只显示前10个
            print(f"  Polygon {i+1}: {len(list(poly.exterior.coords))} 个点, 面积={poly.area:.10f}")
            bounds = poly.bounds
            print(f"    边界: ({bounds[0]:.6f}, {bounds[1]:.6f}) -> ({bounds[2]:.6f}, {bounds[3]:.6f})")

