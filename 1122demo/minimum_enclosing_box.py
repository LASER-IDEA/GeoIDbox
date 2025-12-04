import numpy as np
from shapely.geometry import Polygon, MultiPoint, Point, LineString, mapping
from shapely.ops import transform
import pyproj
from pyproj import CRS, Transformer
from functools import partial
import folium
import json
import os
from shapely.wkt import loads as wkt_loads
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy import ndimage
from scipy.ndimage import binary_erosion, label
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
try:
    import matplotlib.pyplot as plt
    from matplotlib import path as mpath
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ----------------------------
# GCJ-02 to WGS84 转换函数（近似逆向）
# 来源：https://github.com/wandergis/coordTransform_py
# ----------------------------

def gcj02_to_wgs84(lng, lat):
    """
    GCJ-02 to WGS84 (approximate inverse)
    """
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * np.pi
    magic = np.sin(radlat)
    magic = 1 - 0.006693421622965943 * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((6356752.3142 / sqrtmagic) * np.pi)
    dlng = (dlng * 180.0) / (6378137.0 * np.pi / np.cos(radlat))
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat

def out_of_china(lng, lat):
    return not (73.66 < lng < 135.05 and 18.25 < lat < 53.85)

def transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(y * np.pi) + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(y / 12.0 * np.pi) + 320 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
    return ret

def transform_lng(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(x * np.pi) + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(x / 12.0 * np.pi) + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
    return ret

# ----------------------------
# 主函数：生成外扩凸包
# ----------------------------

def create_buffered_convex_hull(gcj02_coords, buffer_distance_m=100, output_format='geojson'):
    """
    输入：gcj02_coords = [(lng1, lat1), (lng2, lat2), ...]
    buffer_distance_m: 外扩距离（米）
    output_format: 'geojson' 或 'wkt'
    返回：GeoJSON dict 或 WKT 字符串（WGS84）
    """
    if len(gcj02_coords) < 3:
        raise ValueError("至少需要3个点才能生成凸包")

    # Step 1: GCJ-02 → WGS84
    wgs84_coords = [gcj02_to_wgs84(lng, lat) for lng, lat in gcj02_coords]

    # Step 2: 自动选择 UTM 投影（基于中心点）
    lons, lats = zip(*wgs84_coords)
    center_lon = np.mean(lons)
    center_lat = np.mean(lats)

    # 计算 UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1
    south_flag = "+south" if center_lat < 0 else ""
    utm_crs = f"+proj=utm +zone={utm_zone} {south_flag} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    wgs84_crs = CRS.from_epsg(4326)
    utm_crs_obj = CRS.from_proj4(utm_crs)

    # Step 3: 投影到 UTM
    project_to_utm = Transformer.from_crs(wgs84_crs, utm_crs_obj, always_xy=True).transform

    # 创建 Shapely MultiPoint 并投影
    multi_point = MultiPoint(wgs84_coords)
    utm_multi_point = transform(project_to_utm, multi_point)

    # Step 4: 凸包 + 缓冲
    convex_hull = utm_multi_point.convex_hull
    buffered_hull = convex_hull.buffer(buffer_distance_m)

    # Step 5: 转回 WGS84
    project_to_wgs84 = Transformer.from_crs(utm_crs_obj, wgs84_crs, always_xy=True).transform
    wgs84_polygon = transform(project_to_wgs84, buffered_hull)

    # Step 6: 输出格式
    if output_format.lower() == 'wkt':
        return wgs84_polygon.wkt
    elif output_format.lower() == 'geojson':
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(wgs84_polygon.exterior.coords)]
            },
            "properties": {}
        }
    else:
        raise ValueError("output_format 必须是 'geojson' 或 'wkt'")


def create_buffered_convex_hull_with_intermediates(gcj02_coords, buffer_distance_m=100):
    """
    返回：
    - wgs84_points: [(lng, lat), ...]
    - convex_hull_wgs84: shapely Polygon
    - buffered_hull_wgs84: shapely Polygon
    """
    if len(gcj02_coords) < 3:
        raise ValueError("至少需要3个点")

    # Step 1: GCJ-02 → WGS84
    wgs84_coords = [gcj02_to_wgs84(lng, lat) for lng, lat in gcj02_coords]

    # Step 2: 自动选择 UTM
    lons, lats = zip(*wgs84_coords)
    center_lon = np.mean(lons)
    center_lat = np.mean(lats)
    utm_zone = int((center_lon + 180) / 6) + 1
    south_flag = "+south" if center_lat < 0 else ""
    utm_crs = f"+proj=utm +zone={utm_zone} {south_flag} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    wgs84_crs = CRS.from_epsg(4326)
    utm_crs_obj = CRS.from_proj4(utm_crs)

    # Step 3: 投影到 UTM
    project_to_utm = Transformer.from_crs(wgs84_crs, utm_crs_obj, always_xy=True).transform
    project_to_wgs84 = Transformer.from_crs(utm_crs_obj, wgs84_crs, always_xy=True).transform

    multi_point = MultiPoint(wgs84_coords)
    utm_multi_point = transform(project_to_utm, multi_point)

    # Step 4: 凸包 & 缓冲
    convex_hull_utm = utm_multi_point.convex_hull
    buffered_hull_utm = convex_hull_utm.buffer(buffer_distance_m)

    # Step 5: 转回 WGS84
    convex_hull_wgs84 = transform(project_to_wgs84, convex_hull_utm)
    buffered_hull_wgs84 = transform(project_to_wgs84, buffered_hull_utm)

    return wgs84_coords, convex_hull_wgs84, buffered_hull_wgs84


# ----------------------------
# 等高线扩展函数
# ----------------------------

def extract_contour_polygons_from_file(filepath):
    """
    从单个文件中提取所有等高线polygon
    返回：list of Polygon objects（按面积排序，从外到内）
    """
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

    # 按面积排序（从大到小，外层先）
    polygons = sorted(polygons, key=lambda p: p.area, reverse=True)

    return polygons


def sample_points_from_polygons(polygons_with_values, points_per_unit_length=1.0):
    """
    从polygon边界采样点，并赋予对应的值

    polygons_with_values: [(polygon, value), ...]
    points_per_unit_length: 每单位长度采样点数（粗略估计）

    返回：(x_coords, y_coords, values) numpy arrays
    """
    x_points = []
    y_points = []
    values = []

    # 确定UTM投影用于采样
    if not polygons_with_values:
        return np.array([]), np.array([]), np.array([])

    # 使用第一个polygon的中心来确定UTM zone
    first_poly = polygons_with_values[0][0]
    center = first_poly.centroid
    center_lon = center.x
    center_lat = center.y

    utm_zone = int((center_lon + 180) / 6) + 1
    south_flag = "+south" if center_lat < 0 else ""
    utm_crs = f"+proj=utm +zone={utm_zone} {south_flag} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    wgs84_crs = CRS.from_epsg(4326)
    utm_crs_obj = CRS.from_proj4(utm_crs)

    project_to_utm = Transformer.from_crs(wgs84_crs, utm_crs_obj, always_xy=True).transform
    project_to_wgs84 = Transformer.from_crs(utm_crs_obj, wgs84_crs, always_xy=True).transform

    for poly, value in polygons_with_values:
        # 投影到UTM进行采样
        poly_utm = transform(project_to_utm, poly)

        # 在外边界上采样点
        exterior_coords = list(poly_utm.exterior.coords)
        if len(exterior_coords) < 2:
            continue

        # 计算总长度
        total_length = 0
        for i in range(len(exterior_coords) - 1):
            p1 = exterior_coords[i]
            p2 = exterior_coords[i + 1]
            total_length += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # 采样点数量
        num_points = max(10, int(total_length * points_per_unit_length))

        # 均匀采样
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # 沿着边界插值
            segment_length = 0
            target_length = t * total_length

            for j in range(len(exterior_coords) - 1):
                p1 = exterior_coords[j]
                p2 = exterior_coords[j + 1]
                seg_len = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

                if segment_length + seg_len >= target_length:
                    # 在这个线段上
                    seg_t = (target_length - segment_length) / seg_len if seg_len > 0 else 0
                    x_utm = p1[0] + seg_t * (p2[0] - p1[0])
                    y_utm = p1[1] + seg_t * (p2[1] - p1[1])

                    # 转回WGS84
                    x_wgs84, y_wgs84 = project_to_wgs84(x_utm, y_utm)

                    x_points.append(x_wgs84)
                    y_points.append(y_wgs84)
                    values.append(value)
                    break

                segment_length += seg_len

        # 也可以在内部采样一些点（重心附近）
        centroid_utm = poly_utm.centroid
        centroid_wgs84_x, centroid_wgs84_y = project_to_wgs84(centroid_utm.x, centroid_utm.y)
        x_points.append(centroid_wgs84_x)
        y_points.append(centroid_wgs84_y)
        values.append(value)

    return np.array(x_points), np.array(y_points), np.array(values)


def regenerate_contours_in_enlarged_polygon(data_folder='data', target_polygon=None,
                                            gcj02_points=None, buffer_distance_m=200,
                                            output_folder='data_extended',
                                            sampling_resolution=200,
                                            interpolation_method='linear',
                                            noise_level=0.1):
    """
    在enlarged polygon内重新生成等高线

    方案：
    1. 从data文件夹加载所有polygon
    2. 按面积大小给每个polygon分配常值（面积大的值小，面积小的值大）+ 随机噪声
    3. 使用gcj02_points转换为wgs84，加上buffer_distance_m，生成enlarged polygon
    4. 在enlarged polygon内生成离散采样点
    5. 对每个采样点，找到它落在哪个polygon里（如果落在多个里，选择面积最小的）
    6. 使用该polygon的值进行插值采样
    7. 重建整个场
    8. 重新提取等高线并输出

    sampling_resolution: 采样网格分辨率（点数）
    interpolation_method: 'linear', 'cubic', 'nearest' - 用于最终插值生成网格
    noise_level: 噪声水平（相对于值差的百分比）
    """
    print("\n" + "="*60)
    print("🗺️  重新生成等高线覆盖区域...")
    print("="*60)

    # Step 1: 加载目标多边形
    if target_polygon is None:
        if gcj02_points is None:
            raise ValueError("必须提供 target_polygon 或 gcj02_points")
        print("\n🎯 生成目标多边形（从锚点）...")
        _, _, target_polygon = create_buffered_convex_hull_with_intermediates(
            gcj02_points, buffer_distance_m
        )
        if target_polygon.is_empty:
            raise ValueError("目标多边形为空")

    print(f"✅ 目标多边形: {target_polygon.geom_type}, 面积={target_polygon.area:.8f}")

    # Step 2: 加载所有polygon并分配值
    if not os.path.exists(data_folder):
        print(f"❌ 数据文件夹 {data_folder} 不存在")
        return {}

    print(f"\n📂 从 {data_folder} 加载所有polygon...")
    all_polygons = []  # [(polygon, area, filename), ...]

    files = sorted([f for f in os.listdir(data_folder) if f.endswith('.txt')])

    for filename in files:
        filepath = os.path.join(data_folder, filename)
        try:
            polygons = extract_contour_polygons_from_file(filepath)
            for poly in polygons:
                all_polygons.append((poly, poly.area, filename))
            print(f"✅ {filename}: {len(polygons)} 个polygon")
        except Exception as e:
            print(f"⚠️  {filename}: 加载失败 - {e}")

    if not all_polygons:
        print("❌ 没有找到有效的polygon数据")
        return {}

    print(f"\n📊 总计: {len(all_polygons)} 个polygon")

    # Step 3: 按面积排序并分配值（面积大的值小，面积小的值大）
    print(f"\n💾 为polygon分配值...")
    all_polygons_sorted = sorted(all_polygons, key=lambda x: x[1], reverse=True)  # 按面积从大到小排序

    # 计算值的范围
    min_area = all_polygons_sorted[-1][1]
    max_area = all_polygons_sorted[0][1]
    area_range = max_area - min_area if max_area > min_area else 1.0

    # 为每个polygon分配值（面积大的值小，面积小的值大）
    polygon_values = {}  # {polygon: (base_value, noise_value)}
    base_values = []

    for idx, (poly, area, filename) in enumerate(all_polygons_sorted):
        # 基础值：面积大的值小（从大到小，值从大到小）
        normalized_area = (area - min_area) / area_range if area_range > 0 else 0.5
        base_value = 100 - normalized_area * 80  # 值范围约20-100，可以根据需要调整

        base_values.append(base_value)
        polygon_values[poly] = (base_value, filename)

    # 计算值之间的最小差值
    if len(base_values) > 1:
        value_diffs = [abs(base_values[i] - base_values[i+1]) for i in range(len(base_values)-1)]
        min_value_diff = min(value_diffs) if value_diffs else 1.0
    else:
        min_value_diff = 1.0

    # 确保噪声小于最小差值的一半
    actual_noise_level = min(noise_level * min_value_diff, min_value_diff * 0.3)

    print(f"   值范围: {min(base_values):.2f} - {max(base_values):.2f}")
    print(f"   最小差值: {min_value_diff:.2f}, 噪声水平: {actual_noise_level:.2f}")

    # 为每个polygon添加噪声
    np.random.seed(42)  # 设置随机种子以保持一致性
    for poly in polygon_values:
        base_value, filename = polygon_values[poly]
        noise = np.random.uniform(-actual_noise_level, actual_noise_level)
        polygon_values[poly] = (base_value + noise, filename)

    # Step 4: 在enlarged polygon内生成采样点
    print(f"\n📐 在目标多边形内生成采样点 (分辨率={sampling_resolution})...")
    target_bounds = target_polygon.bounds
    x_min, y_min, x_max, y_max = target_bounds

    # 创建采样网格
    x_grid = np.linspace(x_min, x_max, sampling_resolution)
    y_grid = np.linspace(y_min, y_max, sampling_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # 只保留在目标多边形内的点
    sampling_points = []
    sampling_values = []

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            x, y = X_grid[i, j], Y_grid[i, j]
            point = Point(x, y)

            if target_polygon.contains(point) or target_polygon.touches(point):
                # 找到这个点落在哪些polygon里
                containing_polygons = []

                for poly, (value, filename) in polygon_values.items():
                    if poly.contains(point) or poly.touches(point):
                        containing_polygons.append((poly, poly.area, value, filename))

                if containing_polygons:
                    # 如果有多个，选择面积最小的
                    containing_polygons.sort(key=lambda x: x[1])  # 按面积从小到大排序
                    selected_poly, _, value, _ = containing_polygons[0]
                    sampling_points.append((x, y))
                    sampling_values.append(value)

    sampling_points = np.array(sampling_points)
    sampling_values = np.array(sampling_values)

    print(f"✅ 生成了 {len(sampling_points)} 个采样点")

    if len(sampling_points) < 3:
        print("❌ 采样点太少，无法重建场")
        return {}

    # Step 5: 重建整个场（在完整网格上插值）
    print(f"\n🔧 重建整个场 (插值方法: {interpolation_method})...")

    # 创建完整的插值网格
    grid_resolution = sampling_resolution
    x_grid_full = np.linspace(x_min, x_max, grid_resolution)
    y_grid_full = np.linspace(y_min, y_max, grid_resolution)
    X_grid_full, Y_grid_full = np.meshgrid(x_grid_full, y_grid_full)

    # 在完整网格上插值
    Z_grid_full = griddata(
        sampling_points,
        sampling_values,
        (X_grid_full.flatten(), Y_grid_full.flatten()),
        method=interpolation_method,
        fill_value=np.nan
    ).reshape(X_grid_full.shape)

    # 创建目标多边形的mask
    target_mask = np.zeros_like(X_grid_full, dtype=bool)
    for i in range(X_grid_full.shape[0]):
        for j in range(X_grid_full.shape[1]):
            target_mask[i, j] = target_polygon.contains(Point(X_grid_full[i, j], Y_grid_full[i, j]))

    # 在mask外的区域设为NaN
    Z_grid_full[~target_mask] = np.nan

    print(f"✅ 场重建完成: {X_grid_full.shape}")
    print(f"   有效值范围: {np.nanmin(Z_grid_full):.2f} - {np.nanmax(Z_grid_full):.2f}")

    # Step 6: 提取等高线
    print(f"\n🎨 提取等高线...")

    # 获取所有唯一的采样值（用于生成等高线层级）
    unique_values = sorted(set(sampling_values))
    print(f"📈 等高线层级: {len(unique_values)} 个 (值范围: {min(unique_values):.2f} - {max(unique_values):.2f})")

    # 按文件分组（根据原始文件）
    files_dict = {}
    for poly, (value, filename) in polygon_values.items():
        if filename not in files_dict:
            files_dict[filename] = []
        files_dict[filename].append(value)

    os.makedirs(output_folder, exist_ok=True)
    extended_results = {}

    # 为每个文件生成等高线
    for filename in sorted(files_dict.keys()):
        file_values = sorted(set(files_dict[filename]))
        print(f"\n📝 处理 {filename}: 目标值 {file_values}")

        contour_polygons = []

        # 为每个目标值生成等高线
        for target_value in file_values:
            print(f"  🎯 生成值={target_value:.2f} 的等高线...")

            try:
                if HAS_SKIMAGE:
                    # 使用skimage.measure.find_contours从2D网格提取等高线
                    contours_found = measure.find_contours(Z_grid_full, target_value)
                    print(f"    ✅ skimage找到 {len(contours_found)} 条等高线")
                elif HAS_MATPLOTLIB:
                    # 使用matplotlib的contour功能
                    fig, ax = plt.subplots(figsize=(1, 1))
                    cs = ax.contour(X_grid_full, Y_grid_full, Z_grid_full, levels=[target_value])
                    plt.close(fig)

                    contours_found = []
                    for collection in cs.collections:
                        for path in collection.get_paths():
                            vertices = path.vertices
                            if len(vertices) >= 3:
                                contours_found.append(vertices)
                    print(f"    ✅ matplotlib找到 {len(contours_found)} 条等高线")
                else:
                    # 回退方法：使用阈值提取区域
                    raise ImportError("No skimage or matplotlib")

                for contour_idx, contour in enumerate(contours_found):
                    if len(contour) < 3:
                        continue

                    contour_coords = []

                    if HAS_SKIMAGE:
                        # skimage返回的是(row, col)，需要转换为(x, y)
                        for point in contour:
                            y_idx, x_idx = point

                            # 更精确的插值
                            if 0 <= x_idx < len(x_grid_full) - 1 and 0 <= y_idx < len(y_grid_full) - 1:
                                x_frac = x_idx - int(x_idx)
                                y_frac = y_idx - int(y_idx)
                                x_coord = x_grid_full[int(x_idx)] * (1 - x_frac) + x_grid_full[int(x_idx) + 1] * x_frac
                                y_coord = y_grid_full[int(y_idx)] * (1 - y_frac) + y_grid_full[int(y_idx) + 1] * y_frac
                            else:
                                x_coord = x_grid_full[min(int(round(x_idx)), len(x_grid_full) - 1)]
                                y_coord = y_grid_full[min(int(round(y_idx)), len(y_grid_full) - 1)]

                            contour_coords.append((x_coord, y_coord))
                    elif HAS_MATPLOTLIB:
                        # matplotlib返回的已经是(x, y)坐标
                        for point in contour:
                            contour_coords.append((point[0], point[1]))

                    if len(contour_coords) >= 3:
                        # 确保闭合
                        if contour_coords[0] != contour_coords[-1]:
                            contour_coords.append(contour_coords[0])

                        try:
                            contour_poly = Polygon(contour_coords)

                            # 确保在目标多边形内
                            intersection = contour_poly.intersection(target_polygon)

                            if isinstance(intersection, Polygon) and not intersection.is_empty:
                                # 检查是否重复（相似的多边形）
                                is_duplicate = False
                                for existing_poly in contour_polygons:
                                    # 检查面积相似度和重叠度
                                    area_ratio = min(intersection.area, existing_poly.area) / max(intersection.area, existing_poly.area)
                                    overlap_ratio = intersection.intersection(existing_poly).area / max(intersection.area, existing_poly.area)
                                    if area_ratio > 0.95 and overlap_ratio > 0.9:
                                        is_duplicate = True
                                        break

                                if not is_duplicate:
                                    contour_polygons.append(intersection)
                                    print(f"      ✅ 添加等高线 {len(contour_polygons)}: {len(contour_coords)} 个点, 面积={intersection.area:.8f}")
                            elif hasattr(intersection, 'geoms'):
                                # MultiPolygon
                                for geom in intersection.geoms:
                                    if isinstance(geom, Polygon) and not geom.is_empty:
                                        is_duplicate = False
                                        for existing_poly in contour_polygons:
                                            area_ratio = min(geom.area, existing_poly.area) / max(geom.area, existing_poly.area)
                                            overlap_ratio = geom.intersection(existing_poly).area / max(geom.area, existing_poly.area)
                                            if area_ratio > 0.95 and overlap_ratio > 0.9:
                                                is_duplicate = True
                                                break
                                        if not is_duplicate:
                                            contour_polygons.append(geom)
                                            print(f"      ✅ 添加等高线 {len(contour_polygons)}: {len(geom.exterior.coords)} 个点, 面积={geom.area:.8f}")
                        except Exception as e:
                            # 如果创建polygon失败，尝试使用凸包
                            try:
                                if len(contour_coords) >= 3:
                                    contour_poly = MultiPoint(contour_coords).convex_hull
                                    if isinstance(contour_poly, Polygon) and not contour_poly.is_empty:
                                        intersection = contour_poly.intersection(target_polygon)
                                        if isinstance(intersection, Polygon) and not intersection.is_empty:
                                            contour_polygons.append(intersection)
                                            print(f"      ✅ 添加等高线(凸包) {len(contour_polygons)}: 面积={intersection.area:.8f}")
                            except:
                                pass

            except (ImportError, Exception) as e:
                # 回退方法：使用连通域分析提取多条等高线
                print(f"    ⚠️  使用回退方法 (原因: {type(e).__name__})")

                threshold = np.nanstd(Z_grid_full) * 0.1 if not np.isnan(Z_grid_full).all() else 1.0

                # 在完整网格上找到符合条件的点
                mask_grid = np.abs(Z_grid_full - target_value) <= threshold
                mask_grid = mask_grid & target_mask  # 确保在目标多边形内
                mask_grid = mask_grid & ~np.isnan(Z_grid_full)  # 排除NaN

                if mask_grid.any():
                    # 使用连通域分析来分离不同的等高线区域
                    # 标记连通域
                    labeled_array, num_features = label(mask_grid)

                    print(f"    📊 找到 {num_features} 个连通域")

                    # 为每个连通域生成等高线
                    for label_id in range(1, num_features + 1):
                        region_mask = (labeled_array == label_id)

                        if region_mask.sum() < 3:  # 至少需要3个点
                            continue

                        # 提取该区域的边界点
                        # 找到区域的边界
                        eroded = binary_erosion(region_mask)
                        boundary = region_mask & ~eroded

                        if boundary.sum() < 3:
                            # 如果边界点太少，使用整个区域
                            boundary = region_mask

                        # 提取坐标
                        contour_x = X_grid_full[boundary]
                        contour_y = Y_grid_full[boundary]

                        if len(contour_x) >= 3:
                            contour_points = list(zip(contour_x, contour_y))

                            try:
                                # 使用凸包
                                contour_poly = MultiPoint(contour_points).convex_hull
                                if isinstance(contour_poly, Polygon) and not contour_poly.is_empty:
                                    # 确保在目标多边形内
                                    intersection = contour_poly.intersection(target_polygon)

                                    if isinstance(intersection, Polygon) and not intersection.is_empty:
                                        # 检查是否重复
                                        is_duplicate = False
                                        for existing_poly in contour_polygons:
                                            area_ratio = min(intersection.area, existing_poly.area) / max(intersection.area, existing_poly.area)
                                            overlap_ratio = intersection.intersection(existing_poly).area / max(intersection.area, existing_poly.area)
                                            if area_ratio > 0.95 and overlap_ratio > 0.9:
                                                is_duplicate = True
                                                break

                                        if not is_duplicate:
                                            contour_polygons.append(intersection)
                                            print(f"      ✅ 添加等高线 {len(contour_polygons)}: {len(contour_points)} 个点, 面积={intersection.area:.8f}")
                                    elif hasattr(intersection, 'geoms'):
                                        for geom in intersection.geoms:
                                            if isinstance(geom, Polygon) and not geom.is_empty:
                                                contour_polygons.append(geom)
                                                print(f"      ✅ 添加等高线 {len(contour_polygons)}: 面积={geom.area:.8f}")
                            except Exception as err:
                                print(f"      ⚠️  处理连通域 {label_id} 失败: {err}")
                                pass

        # 按面积排序（从外到内）
        contour_polygons = sorted(contour_polygons, key=lambda p: p.area, reverse=True)

        # 保存到文件
        if contour_polygons:
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                # 保存所有polygon（每个polygon用分号分隔）
                for i, poly in enumerate(contour_polygons):
                    if i > 0:
                        f.write(';')
                    f.write(poly.wkt)

            extended_results[filename] = contour_polygons
            print(f"✅ {filename}: 生成了 {len(contour_polygons)} 条等高线")
        else:
            print(f"⚠️  {filename}: 未能生成等高线")

    print(f"\n✅ 完成！扩展的等高线已保存到 {output_folder}")
    return extended_results


# ----------------------------
# Error Contour提取函数
# ----------------------------

def extract_error_contours_from_json(target_polygon, json_filepath,
                                     contour_levels=None,
                                     output_wkt_filepath=None):
    """
    从JSON文件中提取指定区域的error contour

    参数:
        target_polygon: shapely.Polygon - 目标区域多边形（WGS84）
        json_filepath: str - JSON文件路径
        contour_levels: int or list - 等高线层级数或具体的层级值列表
        output_wkt_filepath: str - 输出的WKT文件路径（如果为None，则从json文件名生成）

    返回:
        list of Polygon - 提取的等高线polygon列表
    """
    import json
    import glob

    try:
        # 读取JSON文件
        with open(json_filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # 提取数据
        interpolated_grid = json_data.get('interpolated_grid', {})
        grid_x_list = interpolated_grid.get('grid_x', [])
        grid_y_list = interpolated_grid.get('grid_y', [])
        error_field_list = interpolated_grid.get('error_field', [])

        if not grid_x_list or not grid_y_list or not error_field_list:
            print(f"  ⚠️  {os.path.basename(json_filepath)}: 没有有效的插值网格数据")
            return []

        # 转换为numpy数组
        # 根据visualize_demo，grid_x/grid_y可能是二维列表，需要reshape
        try:
            grid_x_data = np.array(grid_x_list)
            grid_y_data = np.array(grid_y_list)
            error_field_data = np.array(error_field_list)

            # 获取唯一值来确定网格尺寸
            if grid_x_data.ndim == 1:
                # 一维数组
                unique_x = np.unique(grid_x_data)
                unique_y = np.unique(grid_y_data)
                grid_width = len(unique_x)
                grid_height = len(unique_y)

                # reshape为二维网格
                grid_x = grid_x_data.reshape((grid_height, grid_width))
                grid_y = grid_y_data.reshape((grid_height, grid_width))
                error_field = error_field_data.reshape((grid_height, grid_width))

            elif grid_x_data.ndim == 2:
                # 二维数组，直接使用
                grid_x = grid_x_data
                grid_y = grid_y_data
                error_field = error_field_data

            else:
                # 尝试flatten后reshape
                grid_x_flat = grid_x_data.flatten()
                grid_y_flat = grid_y_data.flatten()
                error_field_flat = error_field_data.flatten()

                unique_x = np.unique(grid_x_flat)
                unique_y = np.unique(grid_y_flat)
                grid_width = len(unique_x)
                grid_height = len(unique_y)

                grid_x = grid_x_flat.reshape((grid_height, grid_width))
                grid_y = grid_y_flat.reshape((grid_height, grid_width))
                error_field = error_field_flat.reshape((grid_height, grid_width))

        except Exception as e:
            print(f"  ⚠️  {os.path.basename(json_filepath)}: 处理网格数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return []

        if grid_x.shape != error_field.shape or grid_y.shape != error_field.shape:
            print(f"  ⚠️  {os.path.basename(json_filepath)}: 网格数据形状不匹配 - grid_x:{grid_x.shape}, grid_y:{grid_y.shape}, error_field:{error_field.shape}")
            return []

        # 确保在目标多边形内的数据点才参与contour提取
        # 创建mask：标记哪些网格点在目标多边形内
        print(f"    网格尺寸: {grid_x.shape}, 误差范围: [{np.nanmin(error_field):.2f}, {np.nanmax(error_field):.2f}] 米")

        mask = np.zeros_like(error_field, dtype=bool)
        total_points = grid_x.shape[0] * grid_x.shape[1]

        # 优化：批量检查点是否在多边形内（对于大网格）
        bounds = target_polygon.bounds
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                x, y = float(grid_x[i, j]), float(grid_y[i, j])
                # 快速边界框检查
                if (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
                    point = Point(x, y)
                    if target_polygon.contains(point) or target_polygon.touches(point):
                        mask[i, j] = True

        valid_points_count = mask.sum()
        print(f"    目标区域内的网格点: {valid_points_count}/{total_points}")

        if valid_points_count == 0:
            print(f"  ⚠️  {os.path.basename(json_filepath)}: 目标区域与数据区域无重叠")
            return []

        # 将mask外的区域设为NaN
        error_field_masked = error_field.copy().astype(float)
        error_field_masked[~mask] = np.nan

        # 确定等高线层级
        if contour_levels is None:
            # 自动计算层级数
            valid_values = error_field_masked[~np.isnan(error_field_masked)]
            if len(valid_values) > 0:
                min_val = np.nanmin(valid_values)
                max_val = np.nanmax(valid_values)
                contour_levels = np.linspace(min_val, max_val, 20)  # 20个层级
            else:
                return []
        elif isinstance(contour_levels, int):
            # 指定层级数
            valid_values = error_field_masked[~np.isnan(error_field_masked)]
            if len(valid_values) > 0:
                min_val = np.nanmin(valid_values)
                max_val = np.nanmax(valid_values)
                contour_levels = np.linspace(min_val, max_val, contour_levels)
            else:
                return []

        # 提取等高线
        contour_polygons = []

        try:
            if HAS_SKIMAGE:
                # 使用skimage提取等高线
                for level in contour_levels:
                    contours = measure.find_contours(error_field_masked, level)
                    for contour in contours:
                        if len(contour) < 3:
                            continue

                        contour_coords = []
                        for point in contour:
                            y_idx, x_idx = point

                            # 转换为实际坐标
                            # skimage返回的索引是(y_idx, x_idx)，对应网格的行和列
                            if 0 <= x_idx < grid_x.shape[1] - 1 and 0 <= y_idx < grid_x.shape[0] - 1:
                                # 双线性插值获取精确坐标
                                x_frac = x_idx - int(x_idx)
                                y_frac = y_idx - int(y_idx)
                                i0, i1 = int(y_idx), int(y_idx) + 1
                                j0, j1 = int(x_idx), int(x_idx) + 1

                                x_coord = (grid_x[i0, j0] * (1 - x_frac) * (1 - y_frac) +
                                          grid_x[i0, j1] * x_frac * (1 - y_frac) +
                                          grid_x[i1, j0] * (1 - x_frac) * y_frac +
                                          grid_x[i1, j1] * x_frac * y_frac)
                                y_coord = (grid_y[i0, j0] * (1 - x_frac) * (1 - y_frac) +
                                          grid_y[i0, j1] * x_frac * (1 - y_frac) +
                                          grid_y[i1, j0] * (1 - x_frac) * y_frac +
                                          grid_y[i1, j1] * x_frac * y_frac)
                            else:
                                # 边界情况，直接使用最近的网格点
                                i = min(int(round(y_idx)), grid_x.shape[0] - 1)
                                j = min(int(round(x_idx)), grid_x.shape[1] - 1)
                                x_coord = grid_x[i, j]
                                y_coord = grid_y[i, j]

                            contour_coords.append((x_coord, y_coord))

                        if len(contour_coords) >= 3:
                            # 确保闭合
                            if contour_coords[0] != contour_coords[-1]:
                                contour_coords.append(contour_coords[0])

                            try:
                                contour_poly = Polygon(contour_coords)
                                # 确保在目标多边形内
                                intersection = contour_poly.intersection(target_polygon)

                                if isinstance(intersection, Polygon) and not intersection.is_empty:
                                    contour_polygons.append(intersection)
                                elif hasattr(intersection, 'geoms'):
                                    for geom in intersection.geoms:
                                        if isinstance(geom, Polygon) and not geom.is_empty:
                                            contour_polygons.append(geom)
                            except:
                                pass

            elif HAS_MATPLOTLIB:
                # 使用matplotlib提取等高线
                fig, ax = plt.subplots(figsize=(1, 1))
                cs = ax.contour(grid_x, grid_y, error_field_masked, levels=contour_levels)
                plt.close(fig)

                for collection in cs.collections:
                    for path in collection.get_paths():
                        vertices = path.vertices
                        if len(vertices) < 3:
                            continue

                        contour_coords = [(v[0], v[1]) for v in vertices]

                        if len(contour_coords) >= 3:
                            # 确保闭合
                            if contour_coords[0] != contour_coords[-1]:
                                contour_coords.append(contour_coords[0])

                            try:
                                contour_poly = Polygon(contour_coords)
                                # 确保在目标多边形内
                                intersection = contour_poly.intersection(target_polygon)

                                if isinstance(intersection, Polygon) and not intersection.is_empty:
                                    contour_polygons.append(intersection)
                                elif hasattr(intersection, 'geoms'):
                                    for geom in intersection.geoms:
                                        if isinstance(geom, Polygon) and not geom.is_empty:
                                            contour_polygons.append(geom)
                            except:
                                pass
            else:
                # 回退方法：使用阈值提取区域
                for level in contour_levels:
                    threshold = np.nanstd(error_field_masked) * 0.1 if not np.isnan(error_field_masked).all() else 1.0
                    mask_level = np.abs(error_field_masked - level) <= threshold
                    mask_level = mask_level & mask  # 确保在目标多边形内

                    if mask_level.any():
                        # 提取边界点
                        from scipy.ndimage import binary_erosion
                        eroded = binary_erosion(mask_level)
                        boundary = mask_level & ~eroded

                        if boundary.sum() >= 3:
                            contour_x = grid_x[boundary]
                            contour_y = grid_y[boundary]
                            contour_points = list(zip(contour_x, contour_y))

                            if len(contour_points) >= 3:
                                try:
                                    contour_poly = MultiPoint(contour_points).convex_hull
                                    if isinstance(contour_poly, Polygon) and not contour_poly.is_empty:
                                        intersection = contour_poly.intersection(target_polygon)
                                        if isinstance(intersection, Polygon) and not intersection.is_empty:
                                            contour_polygons.append(intersection)
                                except:
                                    pass

        except Exception as e:
            print(f"  ⚠️  提取等高线时出错: {e}")
            return []

        # 按面积排序（从外到内）
        contour_polygons = sorted(contour_polygons, key=lambda p: p.area, reverse=True)

        return contour_polygons

    except Exception as e:
        print(f"  ❌ 处理 {os.path.basename(json_filepath)} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return []


def generate_error_contours_for_region(gcj02_points, buffer_distance_m=200,
                                       json_folder='height_error_visualization_raw_data',
                                       output_folder='error_contours',
                                       contour_levels=20):
    """
    为指定区域生成所有JSON文件的error contour

    参数:
        gcj02_points: list of tuples - GCJ-02坐标点列表
        buffer_distance_m: float - 缓冲区距离（米）
        json_folder: str - JSON文件所在文件夹
        output_folder: str - 输出文件夹
        contour_levels: int or list - 等高线层级数或具体的层级值列表
    """
    print("\n" + "="*60)
    print("🗺️  开始为指定区域生成error contour...")
    print("="*60)

    # Step 1: 生成目标多边形
    print("\n🎯 生成目标多边形（从锚点）...")
    _, _, target_polygon = create_buffered_convex_hull_with_intermediates(
        gcj02_points, buffer_distance_m
    )

    if target_polygon.is_empty:
        raise ValueError("目标多边形为空")

    print(f"✅ 目标多边形: {target_polygon.geom_type}, 面积={target_polygon.area:.8f}")
    bounds = target_polygon.bounds
    print(f"   边界: ({bounds[0]:.6f}, {bounds[1]:.6f}) -> ({bounds[2]:.6f}, {bounds[3]:.6f})")

    # Step 2: 查找所有JSON文件
    if not os.path.exists(json_folder):
        print(f"❌ JSON文件夹 {json_folder} 不存在")
        return

    json_files = sorted([f for f in os.listdir(json_folder)
                        if f.endswith('_visualization_raw_data.json')])

    if not json_files:
        print(f"❌ 在 {json_folder} 中没有找到JSON文件")
        return

    print(f"\n📂 找到 {len(json_files)} 个JSON文件")

    # Step 3: 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # Step 4: 处理每个JSON文件
    success_count = 0

    for json_filename in json_files:
        json_filepath = os.path.join(json_folder, json_filename)

        print(f"\n📄 处理: {json_filename}")

        # 提取等高线
        contour_polygons = extract_error_contours_from_json(
            target_polygon=target_polygon,
            json_filepath=json_filepath,
            contour_levels=contour_levels,
            output_wkt_filepath=None
        )

        if contour_polygons:
            # 生成输出文件名（从JSON文件名提取时间戳）
            # 例如: height_error_20251112_17_visualization_raw_data.json -> error_contour_20251112_17.txt
            base_name = json_filename.replace('_visualization_raw_data.json', '')
            if base_name.startswith('height_error_'):
                base_name = base_name.replace('height_error_', 'error_contour_')

            output_filename = f"{base_name}.txt"
            output_filepath = os.path.join(output_folder, output_filename)

            # 保存为WKT格式（每个polygon用分号分隔）
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for i, poly in enumerate(contour_polygons):
                    if i > 0:
                        f.write(';')
                    f.write(poly.wkt)

            print(f"  ✅ 生成 {len(contour_polygons)} 条等高线 -> {output_filename}")
            success_count += 1
        else:
            print(f"  ⚠️  未生成等高线")

    print(f"\n✅ 完成！成功处理 {success_count}/{len(json_files)} 个文件，结果保存在 {output_folder}")


# ----------------------------
# 数据加载和扩散函数
# ----------------------------

def load_polygons_from_data_folder(data_folder='data'):
    """
    从data文件夹加载所有多边形（WKT格式）
    返回：{filename: shapely.Polygon, ...}
    """
    polygons = {}
    if not os.path.exists(data_folder):
        print(f"⚠️  数据文件夹 {data_folder} 不存在")
        return polygons

    for filename in sorted(os.listdir(data_folder)):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        continue

                    # 文件可能包含多个POLYGON，需要分割处理
                    # 首先尝试解析整个内容
                    try:
                        poly = wkt_loads(content)
                        if isinstance(poly, Polygon) and not poly.is_empty:
                            polygons[filename] = poly
                            print(f"✅ 加载 {filename}: {len(list(poly.exterior.coords))} 个点")
                            continue
                    except:
                        pass

                    # 如果整体解析失败，尝试分割多个POLYGON
                    # 使用括号匹配来找到完整的POLYGON定义
                    polygons_in_file = []
                    i = 0
                    content_len = len(content)

                    while i < content_len:
                        # 查找下一个POLYGON
                        poly_start = content.find('POLYGON', i)
                        if poly_start == -1:
                            break

                        # 从POLYGON开始，找到匹配的闭合括号
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
                            wkt_str = content[poly_start:end_pos].strip()
                            # 移除可能的尾随分号或逗号
                            wkt_str = wkt_str.rstrip(';,').strip()

                            try:
                                poly = wkt_loads(wkt_str)
                                if isinstance(poly, Polygon) and not poly.is_empty:
                                    polygons_in_file.append(poly)
                            except Exception as parse_err:
                                # 如果解析失败，尝试清理字符串
                                # 移除可能的多余字符
                                wkt_str_clean = wkt_str.split(';')[0].split(',POLYGON')[0].strip()
                                try:
                                    poly = wkt_loads(wkt_str_clean)
                                    if isinstance(poly, Polygon) and not poly.is_empty:
                                        polygons_in_file.append(poly)
                                except:
                                    pass

                        i = end_pos if end_pos > poly_start else poly_start + 7

                    # 如果有多个多边形，合并它们或使用第一个
                    if polygons_in_file:
                        # 如果只有一个多边形，直接使用
                        if len(polygons_in_file) == 1:
                            polygons[filename] = polygons_in_file[0]
                        else:
                            # 合并多个多边形：使用所有点的凸包
                            all_points = []
                            for p in polygons_in_file:
                                all_points.extend(list(p.exterior.coords))

                            if all_points and len(all_points) >= 3:
                                union_poly = MultiPoint(all_points).convex_hull
                                if isinstance(union_poly, Polygon) and not union_poly.is_empty:
                                    polygons[filename] = union_poly
                                else:
                                    # fallback: 使用面积最大的多边形
                                    polygons[filename] = max(polygons_in_file, key=lambda p: p.area)
                            else:
                                polygons[filename] = polygons_in_file[0]

                        poly_to_use = polygons[filename]
                        print(f"✅ 加载 {filename}: {len(polygons_in_file)} 个多边形，使用合并结果 {len(list(poly_to_use.exterior.coords))} 个点")
                    else:
                        print(f"⚠️  加载 {filename}: 未找到有效的多边形")

            except Exception as e:
                print(f"⚠️  加载 {filename} 失败: {e}")

    return polygons


def compute_polygon_transformation(original_polygon, target_polygon):
    """
    计算从原始多边形到目标多边形的变换参数
    使用中心对齐的比例缩放方法

    返回：UTM投影变换器和缩放参数
    """
    # 计算中心点和边界框
    orig_centroid = original_polygon.centroid
    target_centroid = target_polygon.centroid

    orig_bounds = original_polygon.bounds  # (minx, miny, maxx, maxy)
    target_bounds = target_polygon.bounds

    # 计算边界框尺寸
    orig_width = orig_bounds[2] - orig_bounds[0]
    orig_height = orig_bounds[3] - orig_bounds[1]
    target_width = target_bounds[2] - target_bounds[0]
    target_height = target_bounds[3] - target_bounds[1]

    # 计算缩放因子（使用平均比例以保持形状）
    scale_x = target_width / orig_width if orig_width > 0 else 1.0
    scale_y = target_height / orig_height if orig_height > 0 else 1.0

    # 使用统一的缩放因子以保持纵横比（更合理）
    # 或者使用各向异性缩放以完全填充目标区域
    # 这里使用统一缩放，基于面积比
    orig_area = original_polygon.area if hasattr(original_polygon, 'area') else orig_width * orig_height
    target_area = target_polygon.area if hasattr(target_polygon, 'area') else target_width * target_height
    uniform_scale = np.sqrt(target_area / orig_area) if orig_area > 0 else 1.0

    # 中心点差（平移量）
    center_shift = (
        target_centroid.x - orig_centroid.x,
        target_centroid.y - orig_centroid.y
    )

    return {
        'original_centroid': orig_centroid,
        'target_centroid': target_centroid,
        'center_shift': center_shift,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'uniform_scale': uniform_scale,
        'original_bounds': orig_bounds,
        'target_bounds': target_bounds
    }


def diffuse_polygon(original_polygon, target_polygon, use_uniform_scale=True):
    """
    将原始多边形扩散到目标多边形区域
    使用中心对齐的比例缩放变换（在UTM投影空间中操作以获得更准确的米级缩放）

    use_uniform_scale: True使用统一缩放保持形状, False使用各向异性缩放填充目标
    """
    if original_polygon.is_empty or target_polygon.is_empty:
        return original_polygon

    # 计算中心点用于确定UTM zone
    orig_centroid = original_polygon.centroid
    center_lon = orig_centroid.x
    center_lat = orig_centroid.y

    # 确定UTM投影
    utm_zone = int((center_lon + 180) / 6) + 1
    south_flag = "+south" if center_lat < 0 else ""
    utm_crs = f"+proj=utm +zone={utm_zone} {south_flag} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    wgs84_crs = CRS.from_epsg(4326)
    utm_crs_obj = CRS.from_proj4(utm_crs)

    # 投影变换器
    project_to_utm = Transformer.from_crs(wgs84_crs, utm_crs_obj, always_xy=True).transform
    project_to_wgs84 = Transformer.from_crs(utm_crs_obj, wgs84_crs, always_xy=True).transform

    # 投影到UTM空间
    orig_poly_utm = transform(project_to_utm, original_polygon)
    target_poly_utm = transform(project_to_utm, target_polygon)

    # 在UTM空间中计算变换参数
    transform_params = compute_polygon_transformation(orig_poly_utm, target_poly_utm)

    orig_centroid_utm = transform_params['original_centroid']
    target_centroid_utm = transform_params['target_centroid']
    scale = transform_params['uniform_scale'] if use_uniform_scale else \
            (transform_params['scale_x'], transform_params['scale_y'])

    def transform_point_utm(x, y):
        # 1. 相对于原始中心平移
        dx = x - orig_centroid_utm.x
        dy = y - orig_centroid_utm.y

        # 2. 缩放
        if use_uniform_scale:
            dx *= scale
            dy *= scale
        else:
            dx *= scale[0]
            dy *= scale[1]

        # 3. 平移到目标中心
        new_x = target_centroid_utm.x + dx
        new_y = target_centroid_utm.y + dy

        return (new_x, new_y)

    # 在UTM空间中转换所有坐标点
    if isinstance(orig_poly_utm, Polygon):
        # 转换外边界（确保闭合）
        exterior_coords_utm = [transform_point_utm(x, y) for x, y in orig_poly_utm.exterior.coords[:-1]]
        if len(exterior_coords_utm) >= 3:
            # 确保闭合
            if exterior_coords_utm[0] != exterior_coords_utm[-1]:
                exterior_coords_utm.append(exterior_coords_utm[0])

        # 转换内边界（holes）
        holes_utm = []
        for interior in orig_poly_utm.interiors:
            hole_coords_utm = [transform_point_utm(x, y) for x, y in interior.coords[:-1]]
            if len(hole_coords_utm) >= 3:
                # 确保闭合
                if hole_coords_utm[0] != hole_coords_utm[-1]:
                    hole_coords_utm.append(hole_coords_utm[0])
                holes_utm.append(hole_coords_utm)

        # 创建UTM空间中的新多边形
        if holes_utm:
            diffused_poly_utm = Polygon(exterior_coords_utm, holes_utm)
        else:
            diffused_poly_utm = Polygon(exterior_coords_utm)

        # 投影回WGS84
        diffused_poly = transform(project_to_wgs84, diffused_poly_utm)

        return diffused_poly
    else:
        return original_polygon


def diffuse_all_polygons(data_folder='data', target_polygon=None, output_folder='data_diffused',
                          gcj02_points=None, buffer_distance_m=200, use_uniform_scale=True):
    """
    加载所有多边形并扩散到目标多边形区域

    target_polygon: 目标多边形（如果不提供，则从gcj02_points生成）
    output_folder: 输出文件夹
    """
    # 加载原始多边形
    print("\n📂 加载原始多边形数据...")
    original_polygons = load_polygons_from_data_folder(data_folder)

    if not original_polygons:
        print("❌ 没有找到有效的多边形数据")
        return {}

    # 获取或生成目标多边形
    if target_polygon is None:
        if gcj02_points is None:
            raise ValueError("必须提供 target_polygon 或 gcj02_points")
        print("\n🎯 生成目标多边形（从锚点）...")
        _, _, target_polygon = create_buffered_convex_hull_with_intermediates(
            gcj02_points, buffer_distance_m
        )
        if target_polygon.is_empty:
            raise ValueError("目标多边形为空")

    # 计算一个参考原始多边形（使用第一个或合并所有）
    # 为了更好的扩散效果，我们计算所有原始多边形的凸包作为参考
    if len(original_polygons) > 1:
        all_points = []
        for poly in original_polygons.values():
            if isinstance(poly, Polygon):
                all_points.extend(list(poly.exterior.coords))
        if all_points:
            reference_polygon = MultiPoint(all_points).convex_hull
        else:
            reference_polygon = list(original_polygons.values())[0]
    else:
        reference_polygon = list(original_polygons.values())[0]

    print(f"\n🔧 计算变换参数...")
    print(f"   参考多边形: {reference_polygon.geom_type}, 面积={reference_polygon.area:.6f}")
    print(f"   目标多边形: {target_polygon.geom_type}, 面积={target_polygon.area:.6f}")

    # 计算变换参数
    transform_params = compute_polygon_transformation(reference_polygon, target_polygon)
    print(f"   缩放因子: {transform_params['uniform_scale']:.4f}")
    print(f"   中心偏移: ({transform_params['center_shift'][0]:.6f}, {transform_params['center_shift'][1]:.6f})")

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 扩散所有多边形
    print(f"\n🔄 扩散多边形...")
    diffused_polygons = {}

    for filename, orig_poly in original_polygons.items():
        try:
            # 扩散多边形
            diffused_poly = diffuse_polygon(orig_poly, target_polygon, use_uniform_scale)

            if not diffused_poly.is_empty:
                diffused_polygons[filename] = diffused_poly

                # 保存为WKT
                output_path = os.path.join(output_folder, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(diffused_poly.wkt)

                print(f"✅ {filename}: {len(list(diffused_poly.exterior.coords))} 个点 -> {output_path}")
        except Exception as e:
            print(f"⚠️  扩散 {filename} 失败: {e}")

    print(f"\n✅ 完成！已扩散 {len(diffused_polygons)}/{len(original_polygons)} 个多边形到 {output_folder}")

    return diffused_polygons


def visualize_on_map(gcj02_coords, buffer_distance_m=100, map_filename="convex_hull_map.html"):
    wgs84_coords, convex_hull, buffered_hull = create_buffered_convex_hull_with_intermediates(
        gcj02_coords, buffer_distance_m
    )

    print(wgs84_coords)
    print(convex_hull)
    print(buffered_hull)

    # save buffered_hull to GeoJSON and WKT (Shapely)
    if buffered_hull is not None and not buffered_hull.is_empty:
        buffered_hull_geojson = {
            "type": "Feature",
            "geometry": mapping(buffered_hull),
            "properties": {}
        }
        buffered_hull_wkt = buffered_hull.wkt
        with open('buffered_hull_geojson.json', 'w') as f:
            json.dump(buffered_hull_geojson, f)
        with open('buffered_hull_wkt.wkt', 'w') as f:
            f.write(buffered_hull_wkt)

        print(f"✅ buffered_hull_geojson saved to buffered_hull_geojson.json")
        print(f"✅ buffered_hull_wkt saved to buffered_hull_wkt.wkt")

    # 地图中心
    center_lat = np.mean([pt[1] for pt in wgs84_coords])
    center_lon = np.mean([pt[0] for pt in wgs84_coords])

    # 创建 folium 地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap"
    )

    # 添加原始点（GCJ-02 转 WGS84 后）
    for i, (lng, lat) in enumerate(wgs84_coords):
        folium.Marker(
            location=[lat, lng],
            popup=f"Point {i+1}: ({lng:.5f}, {lat:.5f})",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    # 添加凸包（蓝色）
    if convex_hull is not None and not convex_hull.is_empty:
        if isinstance(convex_hull, Polygon):
            convex_coords = list(convex_hull.exterior.coords) if convex_hull.exterior else []
            if convex_coords:
                folium.Polygon(
                    locations=[(lat, lng) for lng, lat in convex_coords],
                    color='blue',
                    weight=2,
                    fill=False,
                    popup="Convex Hull"
                ).add_to(m)
        elif isinstance(convex_hull, LineString):
            line_coords = list(convex_hull.coords)
            if line_coords:
                folium.PolyLine(
                    locations=[(lat, lng) for lng, lat in line_coords],
                    color='blue',
                    weight=2,
                    popup="Convex Hull (Line)"
                ).add_to(m)
        elif isinstance(convex_hull, Point):
            folium.CircleMarker(
                location=[convex_hull.y, convex_hull.x],
                radius=4,
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.8,
                popup="Convex Hull (Point)"
            ).add_to(m)
    elif isinstance(convex_hull, LineString):
        line_coords = list(convex_hull.coords)
        folium.PolyLine(
            locations=[(lat, lng) for lng, lat in line_coords],
            color='blue',
            weight=2,
            popup="Convex Hull (Line)"
        ).add_to(m)
    elif isinstance(convex_hull, Point):
        folium.CircleMarker(
            location=[convex_hull.y, convex_hull.x],
            radius=4,
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.8,
            popup="Convex Hull (Point)"
        ).add_to(m)

    # 添加缓冲区（绿色，半透明）
    if buffered_hull is not None and not buffered_hull.is_empty and hasattr(buffered_hull, 'exterior') and buffered_hull.exterior:
        buffered_coords = list(buffered_hull.exterior.coords)
        if buffered_coords:
            folium.Polygon(
                locations=[(lat, lng) for lng, lat in buffered_coords],
                color='green',
                weight=2,
                fill=True,
                fillColor='green',
                fillOpacity=0.3,
                popup=f"Buffered Hull ({buffer_distance_m}m)"
            ).add_to(m)

    # 保存并提示
    m.save(map_filename)
    print(f"✅ 地图已保存为: {map_filename}")
    return m

# ----------------------------
# 示例使用
# ----------------------------

if __name__ == "__main__":
    # 示例：GCJ-02 坐标点（北京附近）
    gcj02_points = [
        (114.060536,22.605118),
        (114.05928,22.605177),
        (114.059356,22.604409),
        (114.0603,22.604385),
        (114.061126,22.602126),
        (114.066018,22.602542),
        (114.062928,22.603929),
        (114.063529,22.603201),
        (114.064505,22.605484),
        (114.066104,22.605326)
    ]

    # 生成并可视化（外扩50米）
    visualize_on_map(gcj02_points, buffer_distance_m=200, map_filename="coverage_area.html")

    # 在enlarged区域重新生成等高线
    # print("\n" + "="*60)
    # print("🗺️  开始在enlarged区域重新生成等高线...")
    # print("="*60)

    # extended_contours = regenerate_contours_in_enlarged_polygon(
    #     data_folder='data',
    #     gcj02_points=gcj02_points,
    #     buffer_distance_m=200,
    #     output_folder='data_extended',
    #     sampling_resolution=200,  # 采样网格分辨率
    #     interpolation_method='linear',  # 'linear', 'cubic', 'nearest'
    #     noise_level=0.1  # 噪声水平（相对于值差的百分比）
    # )

    # if extended_contours:
        # print(f"\n✅ 成功扩展 {len(extended_contours)} 个文件的等高线")

    # 为指定区域生成error contour
    # print("\n" + "="*60)
    # print("📊 开始为指定区域生成error contour...")
    # print("="*60)

    # generate_error_contours_for_region(
    #     gcj02_points=gcj02_points,
    #     buffer_distance_m=200,
    #     json_folder='height_error_visualization_raw_data',
    #     output_folder='error_contours',
    #     contour_levels=20  # 等高线层级数
    # )

    # 可选：同时进行简单的扩散（保持原有方法）
    # print("\n" + "="*60)
    # print("🔄 开始扩散多边形数据...")
    # print("="*60)
    # diffused_polygons = diffuse_all_polygons(
    #     data_folder='data',
    #     gcj02_points=gcj02_points,
    #     buffer_distance_m=200,
    #     output_folder='data_diffused',
    #     use_uniform_scale=True
    # )

    # 生成外扩100米的凸包（WGS84）
    # geojson_result = create_buffered_convex_hull(gcj02_points, buffer_distance_m=100, output_format='geojson')
    # wkt_result = create_buffered_convex_hull(gcj02_points, buffer_distance_m=100, output_format='wkt')

    # print("GeoJSON:")
    # print(geojson_result)
    # print("\nWKT:")
    # print(wkt_result)