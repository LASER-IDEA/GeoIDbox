import json
import numpy as np
import shapely.wkt
from shapely.geometry import Polygon, LinearRing, MultiPolygon
import matplotlib.pyplot as plt
import os
import glob

# 加载裁剪区域 WKT (只需加载一次)
wkt_file = 'buffered_hull_wkt.wkt'
print("正在加载裁剪区域...")
with open(wkt_file, 'r') as f:
    wkt_text = f.read()
clip_poly = shapely.wkt.loads(wkt_text)
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

# IDW 插值函数 (反距离加权)
def simple_idw(x, y, z, xi, yi, power=2):
    xi_flat = xi.flatten()
    yi_flat = yi.flatten()

    # 计算网格点到所有已知点的距离
    # 形状: (网格点数, 已知点数)
    dists = np.sqrt((xi_flat[:, None] - x[None, :])**2 + (yi_flat[:, None] - y[None, :])**2)

    # 避免除以0 (极小值替代)
    dists = np.maximum(dists, 1e-12)

    weights = 1.0 / (dists ** power)

    # 加权平均
    z_interp = np.sum(weights * z[None, :], axis=1) / np.sum(weights, axis=1)

    return z_interp.reshape(xi.shape)

def convert_path_to_polygons(path):
    # 将 matplotlib path 转换为 shapely polygons
    polys = []
    verts_list = path.to_polygons()

    exteriors = []
    holes = []

    for verts in verts_list:
        if len(verts) < 3: continue
        ring = LinearRing(verts)
        # matplotlib规则: 逆时针为外环，顺时针为孔洞
        if ring.is_ccw:
            exteriors.append(ring)
        else:
            holes.append(ring)

    if not exteriors: return []

    if len(exteriors) == 1:
        p = Polygon(exteriors[0], holes)
        if not p.is_valid: p = p.buffer(0)
        polys.append(p)
    else:
        # 如果有多个外环，需判断孔洞属于哪个外环
        for ext in exteriors:
            pext = Polygon(ext)
            assigned_holes = [h for h in holes if pext.contains(Polygon(h))]
            p = Polygon(ext, assigned_holes)
            if not p.is_valid: p = p.buffer(0)
            polys.append(p)
    return polys

def process_json_file(json_file_path):
    """处理单个JSON文件并生成WKT输出"""
    print(f"\n正在处理: {json_file_path}")

    # 1. 加载数据
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # 提取原始的9个点 (比使用预插值的grid更准确)
    points = json_data['original_data']
    points_lon = np.array([p['longitude'] for p in points])
    points_lat = np.array([p['latitude'] for p in points])
    points_z = np.array([p['height_error'] for p in points])

    # 2. 设置覆盖 WKT 区域的新网格
    minx, miny, maxx, maxy = clip_poly.bounds
    # 增加一点 buffer 确保覆盖边缘
    padding_x = (maxx - minx) * 0.1
    padding_y = (maxy - miny) * 0.1
    grid_minx, grid_maxx = minx - padding_x, maxx + padding_x
    grid_miny, grid_maxy = miny - padding_y, maxy + padding_y

    # 设置分辨率 (如 500x500)
    res_x = 500
    res_y = int(res_x * (grid_maxy - grid_miny) / (grid_maxx - grid_minx))
    xi = np.linspace(grid_minx, grid_maxx, res_x)
    yi = np.linspace(grid_miny, grid_maxy, res_y)
    XI, YI = np.meshgrid(xi, yi)

    print("正在进行IDW插值扩充...")
    ZI = simple_idw(points_lon, points_lat, points_z, XI, YI, power=2)
    print(f"扩充后数据范围: {np.min(ZI):.2f} 到 {np.max(ZI):.2f}")

    # 4. 生成等值面 (Contourf)
    # 设定轮廓层级，例如每隔2米
    levels = np.arange(np.floor(np.min(ZI)), np.ceil(np.max(ZI)) + 1, 2.0)
    print(f"轮廓层级: {levels}")

    fig, ax = plt.subplots()
    cs = ax.contourf(XI, YI, ZI, levels=levels)
    plt.close(fig)  # 关闭图形以释放内存

    # 5. 提取并裁剪多边形
    output_polygons = []
    print("正在提取轮廓并裁剪...")

    for collection in cs.collections:
        for path in collection.get_paths():
            polys = convert_path_to_polygons(path)
            for p in polys:
                if not p.intersects(clip_poly):
                    continue

                # 执行裁剪 intersection
                try:
                    clipped = p.intersection(clip_poly)

                    if not clipped.is_empty:
                        if isinstance(clipped, Polygon):
                            output_polygons.append(clipped)
                        elif isinstance(clipped, MultiPolygon):
                            output_polygons.extend(clipped.geoms)
                        elif clipped.geom_type == 'GeometryCollection':
                            for g in clipped.geoms:
                                if isinstance(g, Polygon):
                                    output_polygons.append(g)
                except Exception as e:
                    print(f"裁剪错误: {e}")

    # 6. 输出到文件
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_filename = os.path.join(output_dir, f"{base_name}.wkt")
    with open(output_filename, 'w') as f:
        for poly in output_polygons:
            f.write(poly.wkt + '\n')

    print(f"完成。生成了 {len(output_polygons)} 个多边形，已保存至 {output_filename}")
    return len(output_polygons)

# 主程序：处理所有JSON文件
if __name__ == "__main__":
    total_polygons = 0
    for json_file in json_files:
        try:
            polygon_count = process_json_file(json_file)
            total_polygons += polygon_count
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n所有文件处理完成！共处理 {len(json_files)} 个文件，生成 {total_polygons} 个多边形")