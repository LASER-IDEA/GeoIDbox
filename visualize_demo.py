def visualize_height_error_json_files(pattern='height_error_*.json'):
    """
    遍历并可视化保存的高度误差JSON文件

    参数:
        pattern: str - 文件匹配模式
    """
    import glob
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # 查找所有匹配的JSON文件
    json_files = glob.glob(pattern)

    if not json_files:
        print("未找到任何匹配的JSON文件")
        return

    print(f" 找到 {len(json_files)} 个JSON文件")

    # 按文件名排序
    json_files.sort()

    # 为每个文件创建可视化
    for i, file_path in enumerate(json_files):
        try:
            print(f"\n正在处理文件: {file_path}")

            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 检查文件内容
            metadata = json_data.get('metadata', {})
            original_data = json_data.get('original_data', [])
            interpolated_grid = json_data.get('interpolated_grid', {})

            print(f"   原始数据点数量: {len(original_data)}")
            print(f"    网格尺寸: {metadata.get('grid_size', 'Unknown')}")

            # 提取插值网格数据
            grid_x_data = np.array(interpolated_grid.get('grid_x', []))
            grid_y_data = np.array(interpolated_grid.get('grid_y', []))
            error_field_data = np.array(interpolated_grid.get('error_field', []))

            if grid_x_data.size == 0 or grid_y_data.size == 0 or error_field_data.size == 0:
                print(f"   文件中没有有效的插值网格数据")
                continue

            # 重构网格
            # grid_x_data 和 grid_y_data 是一维数组，需要重构为二维网格
            unique_x = np.unique(grid_x_data)
            unique_y = np.unique(grid_y_data)

            # 创建网格坐标矩阵
            xx = grid_x_data.reshape((len(unique_y), len(unique_x)))
            yy = grid_y_data.reshape((len(unique_y), len(unique_x)))
            error_field = error_field_data.reshape((len(unique_y), len(unique_x)))

            print(f"    重构网格尺寸: {xx.shape}")
            print(f"    误差场范围: [{np.nanmin(error_field):.2f}, {np.nanmax(error_field):.2f}] 米")

            # 创建图形
            fig, ax = plt.subplots(figsize=(12, 10))

            # 绘制误差场
            contour_levels = 20
            contour = ax.contour(xx, yy, error_field, levels=contour_levels, colors='black', linewidths=0.5, alpha=0.7)
            contourf = ax.contourf(xx, yy, error_field, levels=contour_levels, cmap='RdBu_r', alpha=0.7)

            # 添加等值线标签
            ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')

            # 绘制原始数据点
            if original_data:
                longitudes = [point['longitude'] for point in original_data]
                latitudes = [point['latitude'] for point in original_data]
                errors = [point['height_error'] for point in original_data]

                scatter = ax.scatter(longitudes, latitudes, c=errors, s=30, marker='o',
                                   cmap='RdBu_r', edgecolors='black', linewidth=0.5, zorder=5)

                # 添加颜色条
                scatter_cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
                scatter_cbar.set_label('Original Height Error (m)', fontsize=10, fontweight='bold')

            # 添加误差场颜色条
            cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
            cbar.set_label('Interpolated Height Error (m)', fontsize=12, fontweight='bold')

            # 设置坐标轴标签
            ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')

            # 设置坐标轴刻度
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

            # 设置标题
            title = f'Height Error Interpolation Field\n{file_path}\n'
            title += f'Original Points: {len(original_data)}, Grid: {xx.shape[1]}x{xx.shape[0]}'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图像
            output_image = file_path.replace('.json', '_reconstructed_visualization.png')
            try:
                fig.savefig(output_image, dpi=300, bbox_inches='tight')
                print(f"重构可视化图像已保存到: {output_image}")
            except Exception as e:
                print(f" 保存图像时出错: {e}")

            # 显示图形（在支持图形界面的环境中）
            plt.show()
            plt.close(fig)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n 完成所有JSON文件的可视化处理")


def visualize_all_height_error_json_files():
    """
    可视化所有高度误差JSON文件的简化函数
    """
    print("开始可视化所有高度误差JSON文件...")
    visualize_height_error_json_files('./height_error_visualization_raw_data/height_error_*_visualization_raw_data.json')
    # visualize_height_error_json_files('test/height_error_*.json')


# 在文件末尾添加调用
visualize_all_height_error_json_files()