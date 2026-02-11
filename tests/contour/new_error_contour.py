import json
import numpy as np
import matplotlib.pyplot as plt
import shapely.wkt
from shapely.geometry import Polygon, LinearRing
from matplotlib.path import Path
import os
import glob

def process_json_file(json_file_path, clip_poly, output_dir):
    """处理单个JSON文件并生成WKT输出"""
    print(f"\n正在处理: {json_file_path}")

    # 1. 加载数据
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    grid_x = np.array(data['interpolated_grid']['grid_x'])
    grid_y = np.array(data['interpolated_grid']['grid_y'])
    grid_z = np.array(data['interpolated_grid']['error_field'])

    print(f"网格尺寸: {grid_z.shape}, 数据范围: {np.nanmin(grid_z):.2f} 到 {np.nanmax(grid_z):.2f}")

    # 2. 生成等值面 (Contourf)
    levels = np.arange(np.floor(np.nanmin(grid_z)), np.ceil(np.nanmax(grid_z)) + 1, 2.0)
    print(f"生成等值线层级: {levels}")

    # 使用 matplotlib 计算等值面
    fig, ax = plt.subplots()
    cs = ax.contourf(grid_x, grid_y, grid_z, levels=levels)
    plt.close(fig)  # 关闭图形以释放内存

    # 3. 提取多边形并裁剪
    output_polygons = []
    print("正在提取并裁剪多边形...")

    for collection in cs.collections:
        for path in collection.get_paths():
            all_rings = []
            for verts in path.to_polygons():
                if len(verts) < 3: continue
                all_rings.append(LinearRing(verts))

            if not all_rings: continue

            # 分离外环和内环
            exteriors = [r for r in all_rings if r.is_ccw]
            holes = [r for r in all_rings if not r.is_ccw]

            # 构建Shapely多边形
            temp_polys = []

            if len(exteriors) == 1:
                poly = Polygon(exteriors[0], holes)
                if not poly.is_valid: poly = poly.buffer(0)
                temp_polys.append(poly)
            else:
                for ext in exteriors:
                    poly_ext = Polygon(ext)
                    my_holes = []
                    for h in holes:
                        if poly_ext.contains(Polygon(h)):
                            my_holes.append(h)
                    poly = Polygon(ext, my_holes)
                    if not poly.is_valid: poly = poly.buffer(0)
                    temp_polys.append(poly)

            # 与目标区域(WKT)进行裁剪 (Intersection)
            for poly in temp_polys:
                try:
                    if not poly.intersects(clip_poly):
                        continue

                    result = poly.intersection(clip_poly)

                    if result.is_empty:
                        continue

                    # 处理结果可能是 MultiPolygon 或 GeometryCollection 的情况
                    if result.geom_type == 'Polygon':
                        output_polygons.append(result)
                    elif result.geom_type == 'MultiPolygon':
                        output_polygons.extend(result.geoms)
                    elif result.geom_type == 'GeometryCollection':
                        for g in result.geoms:
                            if g.geom_type == 'Polygon':
                                output_polygons.append(g)
                except Exception as e:
                    print(f"裁剪多边形时出错: {e}")
                    continue

    # 4. 输出结果到 WKT 文件
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_filename = os.path.join(output_dir, f"{base_name}.wkt")

    with open(output_filename, 'w') as f:
        for poly in output_polygons:
            f.write(poly.wkt + '\n')

    print(f"处理完成。共生成 {len(output_polygons)} 个多边形，已保存至 {output_filename}")
    return len(output_polygons)

# 主程序
if __name__ == "__main__":
    # 加载WKT裁剪区域（只需加载一次）
    print("正在加载裁剪区域...")
    with open('buffered_hull_wkt.wkt', 'r') as f:
        wkt_text = f.read()
    clip_poly = shapely.wkt.loads(wkt_text)

    # 确保裁剪区域有效
    if not clip_poly.is_valid:
        clip_poly = clip_poly.buffer(0)

    # 创建输出文件夹
    output_dir = 'output_contour_wkt'
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出文件夹: {output_dir}")

    # 获取所有JSON文件
    json_folder = 'height_error_visualization_raw_data'
    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    json_files.sort()  # 按文件名排序

    print(f"\n找到 {len(json_files)} 个JSON文件")

    # 处理每个JSON文件
    total_polygons = 0
    for json_file in json_files:
        try:
            polygon_count = process_json_file(json_file, clip_poly, output_dir)
            total_polygons += polygon_count
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            continue

    print(f"\n所有文件处理完成！共处理 {len(json_files)} 个文件，生成 {total_polygons} 个多边形")