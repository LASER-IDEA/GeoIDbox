import cdsapi

# --- 配置区域 ---
# 假设无人机在深圳飞行，时间是 2024年1月15日
# 请根据您下载的无人机日志的实际时间和位置修改这里！
# TARGET_DATE = '2024-01-15'
# TARGET_TIME = ['08:00', '09:00', '10:00'] # UTC时间，根据飞行时段选择
# AREA = [22.7, 113.8, 22.4, 114.2] # North, West, South, East (深圳周边)
TARGET_DATE = '2024-11-24'
TARGET_TIME = '15:53'
# -10KM ~ 10KM, convert to wgs84, roughly divided by earth radius and comvert to degrees
# 10000km / 6371000m = 0.0015708333333333333
# 0.0015708333333333333 * 180 / np.pi = 0.08993201773856615
# 0.08993201773856615 degrees
# 22.9109896,120.312833
# 22.9109896 - 0.08993201773856615 = 22.821057582261434
# 120.312833 - 0.08993201773856615 = 120.22290098226143
# 22.821057582261434,120.40276501773857
# 22.9109896 + 0.08993201773856615 = 23.000921617738566
# 120.312833 + 0.08993201773856615 = 120.40276501773857
AREA = [22.821057582261434,120.22290098226143,23.000921617738566,120.40276501773857]

def download_era5_pressure_levels():
    """
    下载指定时间和区域的ERA5气压层数据 (用于构建垂直剖面)。
    """
    c = cdsapi.Client()
    output_file = f'era5_pl_{TARGET_DATE}.nc'

    print(f"开始下载 ERA5 数据到 {output_file} ... 这可能需要几分钟。")

    # 我们下载不同气压层的温度、比湿和位势高度
    # 重点关注低空层 (例如 1000hPa 到 850hPa)
    # Round time to nearest hour if needed (ERA5 data is hourly)
    if isinstance(TARGET_TIME, str) and ':' in TARGET_TIME:
        parts = TARGET_TIME.split(':')
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        # Round to nearest hour
        if minute >= 30:
            hour = (hour + 1) % 24
        rounded_time = f'{hour:02d}:00'
        if rounded_time != TARGET_TIME:
            print(f"警告: 时间 {TARGET_TIME} 已四舍五入到 {rounded_time} (ERA5 数据为小时级别)")
            time_to_use = rounded_time
        else:
            time_to_use = TARGET_TIME
    else:
        time_to_use = TARGET_TIME

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'data_format': 'netcdf', # 使用 NetCDF 格式方便处理多维数据 (已更新为 data_format)
            'variable': [
                'geopotential', 'specific_humidity', 'temperature',
            ],
            'pressure_level': [
                '50', '100', '150',
                '200', '250', '300',
                '350', '400', '450',
                '500', '550', '600',
                '650', '700', '750',
                '800', '850', '900',
                '950', '1000',
            ],
            'year': TARGET_DATE.split('-')[0],
            'month': TARGET_DATE.split('-')[1],
            'day': TARGET_DATE.split('-')[2],
            'time': time_to_use,
            'area': AREA,
        },
        output_file)
    print("下载完成!")

if __name__ == "__main__":
    # 确保您已经配置好了 .cdsapirc 文件
    download_era5_pressure_levels() # 取消注释并运行
    # print("请先配置 CDS API Key，并修改脚本中的日期和区域，然后再运行。")