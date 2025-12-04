## 高度盒子

### todo
#### 从现有的一期数据中能挖出什么？
这批数据的核心价值在于：它是一个高密度的城市传感器网络阵列。虽然单个传感器的绝对垂直精度不高（普通GNSS MSL），但它们作为一个整体，可以揭示出城市小尺度的气象动力学特征。
1. “静态真值”恢复 (The Static Truth Recovery)
   痛点： 您提到的 GNSS MSL 高度是单点定位结果，跳动极大。
   GNSS 提供的 N 值通常来自接收机内部低精度的查找表，不可全信。
   利用盒子位置固定的特性，进行长时平均。对每个盒子一周的经纬度 $(Lat, Lon)$ 求平均，得到一个高可信的水平坐标。
   基于此固定坐标，使用外部高精度模型 EGM2008 计算该点的大地水准面差距 $N_{precise}$。
   对 GNSS 报告的椭球高（如果能获取）进行长时平均，得到 $\bar{h}_{HAE}$。
   如果只能获取 MSL，则尝试恢复 HAE：$\bar{h}_{HAE} \approx \overline{MSL}_{gnss} + N_{internal\_reported}$。
   产出： 10个高精度的固定 3D 坐标锚点。这是后续一切分析的基石。

2. 城市微气象事件捕捉 (Urban Micro-weather Events Detection)
   思路： 在1km²范围内，如果10个盒子的气压同时下降，那是天气系统（宏观）；
   如果只有两三个相邻的盒子气压剧烈波动，那是局地扰动（微观）。
   挖掘动作：相关性分析： 计算任意两个盒子气压时间序列的互相关系数。
   高相关性意味着它们处于同一气团中。
   阵风/建筑尾流探测： 利用 1Hz 的高频气压数据，计算气压的变化率 $\frac{dp}{dt}$。
   城市建筑间的“狭管效应”或阵风会导致瞬间的局部气压骤变。识别出这些异常时段，这对于无人机飞行安全至关重要。

3. 温度-气压耦合分析 (T-P Coupling in Urban Canyons)
   思路： 城市热岛效应导致局部温度不均匀。分析温度变化如何滞后或超前影响局部气压。
   挖掘动作： 建立局部温度梯度 $\Delta T$ 与气压残差 $\Delta P$ 之间的统计关系模型。这可以作为先验知识输入到未来的神经场中。

### AESM experiment
Altitude Error Statistical Modeling (AESM) is the current production workflow for quantifying how well the barometer tracks GNSS vertical truth across PX4 logs.

#### data preparation
```bash
!python -m pip uninstall -y pip
!sudo apt-get update -y
!sudo apt-get install python3.7
!sudo update-alternatives --remove python /usr/local/bin/python
!sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python3.7
!wget https://bootstrap.pypa.io/get-pip.py
!python get-pip.py
!python --version
!sudo apt-get install sqlite3 libfftw3-bin libfftw3-dev
!sudo apt-get install libatlas3-base
!pip install bokeh jinja2 jupyter pyfftw pylint pyulog>=1.1 requests scipy>=1.8.1 simplekml smopy pycryptodome>=3.18
```

```python
#! /usr/bin/env python3
""" Script to download public logs """

import os
import glob
import argparse
import json
import datetime
import sys
import time
import shutil
from google.colab import drive

# --- Start of fix for ModuleNotFoundError: No module named 'plot_app' ---
# Clone the repository if not already cloned
if not os.path.exists('flight_review'):
    print("Cloning PX4/flight_review repository...")
    os.system('git clone https://github.com/PX4/flight_review.git')
    print("Repository cloned.")

# Add the 'app' directory from the cloned repository to Python's path
flight_review_app_path = os.path.abspath('flight_review/app')
if flight_review_app_path not in sys.path:
    sys.path.insert(0, flight_review_app_path)
    print(f"Added {flight_review_app_path} to sys.path")
# --- End of fix ---

import requests

from plot_app.config_tables import *


def get_arguments():
    """ Get parsed CLI arguments """
    parser = argparse.ArgumentParser(description='Python script for downloading public logs '
                                                 'from the PX4/flight_review database.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max-num', '-n', type=int, default=-1,
                        help='Maximum number of files to download that match the search criteria. '
                             'Default: download all files.')
    parser.add_argument('-d', '--download-folder', type=str, default="data/downloaded/",
                        help='The folder to store the downloaded logfiles.')
    parser.add_argument('--print', action='store_true', dest="print_entries",
                        help='Whether to only print (not download) the database entries.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Whether to overwrite already existing files in download folder.')
    parser.add_argument('--db-info-api', type=str, default="https://review.px4.io/dbinfo",
                        help='The url at which the server provides the dbinfo API.')
    parser.add_argument('--download-api', type=str, default="https://review.px4.io/download",
                        help='The url at which the server provides the download API.')
    parser.add_argument('--mav-type', type=str, default=None, nargs='+',
                        help='Filter logs by mav type (case insensitive). Specifying multiple '
                             'mav types is possible. e.g. Quadrotor, Hexarotor')
    parser.add_argument('--flight-modes', type=str, default=None, nargs='+',
                        help='Filter logs by flight modes. If multiple are provided, the log must '
                             'contain all modes. e.g. Mission')
    parser.add_argument('--error-labels', default=None, nargs='+', type=str,
                        help='Filter logs by error labels. If multiple are provided, the log must '
                             'contain all labels. e.g. Vibration')
    parser.add_argument('--rating', default=None, type=str, nargs='+',
                        help='Filter logs by rating. e.g. Good')
    parser.add_argument('--uuid', default=None, type=str, nargs='+',
                        help='Filter logs by a particular vehicle uuid. e.g. 0123456789')
    parser.add_argument('--log-id', default=None, type=str, nargs='+',
                        help='Filter logs by a particular log id')
    parser.add_argument('--vehicle-name', default=None, type=str,
                        help='Filter logs by a particular vehicle name.')
    parser.add_argument('--airframe-name', default=None, type=str,
                        help='Filter logs by a particular airframe name. e.g. Generic Quadrotor X')
    parser.add_argument('--airframe-type', default=None, type=str,
                        help='Filter logs by a particular airframe type. e.g. Quadrotor X')
    parser.add_argument('--latest-per-vehicle', action='store_true', dest="latest_per_vehicle",
                        help='Download only the latest log (by date) for each ' \
                        'unique vehicle (uuid).')
    parser.add_argument('--source', default=None, type=str,
                        help='The source of the log upload. e.g. ["webui", "CI"]')
    parser.add_argument('--git-hash', default=None, type=str,
                        help='The git hash of the PX4 Firmware version.')
    parser.add_argument('--min-flight-duration', type=int, default=0, # New argument for minimum flight duration
                        help='Filter logs by minimum flight duration in seconds.')
    return parser.parse_args([]) # Modified to ignore kernel arguments


def flight_modes_to_ids(flight_modes):
    """
    returns a list of mode ids for a list of mode labels
    """
    flight_ids = []
    for i,value in flight_modes_table.items():
        if value[0] in flight_modes:
            flight_ids.append(i)
    return flight_ids


def error_labels_to_ids(error_labels):
    """
    returns a list of error ids for a list of error labels
    """
    error_id_table = {label: id for id, label in error_labels_table.items()}
    error_ids = [error_id_table[error_label] for error_label in error_labels]
    return error_ids


def main():
    """ main script entry point """
    args = get_arguments()

    # --- Google Drive Configuration ---
    # Mount Google Drive
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')

    # Define folders
    local_temp_folder = "px4_logs"
    drive_folder = "/content/drive/MyDrive/px4_logs"

    # Override arguments for the current user request
    args.max_num = 500
    args.min_flight_duration = 60
    args.download_folder = drive_folder # Set download folder to Google Drive

    # Move existing local logs to Drive if they exist (to avoid re-downloading)
    if os.path.exists(local_temp_folder):
        if not os.path.exists(drive_folder):
            print(f"Creating directory {drive_folder} on Google Drive...")
            os.makedirs(drive_folder)

        print(f"Checking for existing logs in local folder '{local_temp_folder}' to move to Drive...")
        moved_count = 0
        for file_path in glob.glob(os.path.join(local_temp_folder, "*.ulg")):
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(drive_folder, file_name)
            if not os.path.exists(dest_path):
                shutil.move(file_path, dest_path)
                moved_count += 1
        if moved_count > 0:
            print(f"Moved {moved_count} existing logs to {drive_folder}.")
        else:
            print("No new local logs to move.")
    # ----------------------------------

    try:
        # the db_info_api sends a json file with a list of all public database entries
        db_entries_list = requests.get(url=args.db_info_api, timeout=5*60).json()
    except Exception as e:
        print(f"Server request failed: {e}")
        raise

    if args.print_entries:
        # only print the json output without downloading logs
        print(json.dumps(db_entries_list, indent=4, sort_keys=True))

    else:
        if not os.path.isdir(args.download_folder): # returns true if path is an existing directory
            print("creating download directory " + args.download_folder)
            os.makedirs(args.download_folder)

        # find already existing logs in download folder
        logfile_pattern = os.path.join(os.path.abspath(args.download_folder), "*.ulg")
        logfiles = glob.glob(os.path.join(os.getcwd(), logfile_pattern))
        logids = frozenset(os.path.splitext(os.path.basename(f))[0] for f in logfiles)

        # filter for mav types
        if args.mav_type is not None:
            mav = [mav_type.lower() for mav_type in args.mav_type]
            db_entries_list = [entry for entry in db_entries_list
                               if entry["mav_type"].lower() in mav]

        # filter for rating
        if args.rating is not None:
            rate = [rating.lower() for rating in args.rating]
            db_entries_list = [entry for entry in db_entries_list
                               if entry["rating"].lower() in rate]

        # filter for error labels
        if args.error_labels is not None:
            err_labels = error_labels_to_ids(args.error_labels)
            db_entries_list = [entry for entry in db_entries_list
                               if set(err_labels).issubset(set(entry["error_labels"]))]
            # compares numbers, must contain all

        # filter for flight modes
        if args.flight_modes is not None:
            modes = flight_modes_to_ids(args.flight_modes)
            db_entries_list = [entry for entry in db_entries_list
                               if set(modes).issubset(set(entry["flight_modes"]))]
            # compares numbers, must contain all

        # filter for vehicle uuid
        if args.uuid is not None:
            db_entries_list = [
                entry for entry in db_entries_list if entry['vehicle_uuid'] in args.uuid]


        # filter log_id
        if args.log_id is not None:
            arg_log_ids_without_dashes = [log_id.replace("-", "") for log_id in args.log_id]
            db_entries_list = [
                entry for entry in db_entries_list
                if entry['log_id'].replace("-", "") in arg_log_ids_without_dashes]

        # filter for vehicle name
        if args.vehicle_name is not None:
            db_entries_list = [
                entry for entry in db_entries_list if entry['vehicle_name'] == args.vehicle_name]

        # filter for airframe name
        if args.airframe_name is not None:
            db_entries_list = [
                entry for entry in db_entries_list if entry['airframe_name'] == args.airframe_name]

        # filter for airframe type
        if args.airframe_type is not None:
            db_entries_list = [
                entry for entry in db_entries_list if entry['airframe_type'] == args.airframe_type]

        if args.latest_per_vehicle:
            # find latest log_date for all different vehicles
            uuids = {}
            for entry in db_entries_list:
                if 'vehicle_uuid' in entry:
                    uuid = entry['vehicle_uuid']
                    date = datetime.datetime.strptime(entry['log_date'], '%Y-%m-%d')
                    if uuid in uuids:
                        if date > uuids[uuid]:
                            uuids[uuid] = date
                    else:
                        uuids[uuid] = date
            # filter: use the latest log for each vehicle
            db_entries_list_filtered = []
            added_uuids = set()
            for entry in db_entries_list:
                if 'vehicle_uuid' in entry:
                    date = datetime.datetime.strptime(entry['log_date'], '%Y-%m-%d')
                    uuid = entry['vehicle_uuid']
                    if date == uuids[entry['vehicle_uuid']] and not uuid in added_uuids:
                        db_entries_list_filtered.append(entry)
                        added_uuids.add(uuid)
            db_entries_list = db_entries_list_filtered

        if args.source is not None:
            db_entries_list = [
                entry for entry in db_entries_list
                if 'source' in entry and entry['source'] == args.source]

        if args.git_hash is not None:
            db_entries_list = [
                entry for entry in db_entries_list if entry['ver_sw'] == args.git_hash]

        # Filter for minimum flight duration
        if args.min_flight_duration > 0:
            initial_count = len(db_entries_list)
            db_entries_list = [
                entry for entry in db_entries_list
                if 'duration_s' in entry and entry['duration_s'] >= args.min_flight_duration]
            print(f"Filtered {initial_count - len(db_entries_list)} logs with duration less than {args.min_flight_duration} seconds.")


        # sort list order to first download the newest log files
        db_entries_list = sorted(
            db_entries_list,
            key=lambda x: datetime.datetime.strptime(x['log_date'], '%Y-%m-%d'),
            reverse=True)

        # set number of files to download
        n_en = len(db_entries_list)
        if args.max_num > 0:
            n_en = min(n_en, args.max_num)
        n_downloaded = 0
        n_skipped = 0

        for i in range(n_en):
            entry_id = db_entries_list[i]['log_id']

            num_tries = 0
            for num_tries in range(100):
                try:
                    if args.overwrite or entry_id not in logids:

                        file_path = os.path.join(args.download_folder, entry_id + ".ulg")

                        print('downloading {:}/{:} ({:})'.format(i + 1, n_en, entry_id))
                        request = requests.get(url=args.download_api +
                                               "?log=" + entry_id, stream=True,
                                               timeout=10*60)
                        with open(file_path, 'wb') as log_file:
                            for chunk in request.iter_content(chunk_size=1024):
                                if chunk:  # filter out keep-alive new chunks
                                    log_file.write(chunk)
                        n_downloaded += 1
                    else:
                        n_skipped += 1
                    break
                except Exception as ex:
                    print(ex)
                    print('Waiting for 30 seconds to retry')
                    time.sleep(30)
            if num_tries == 99:
                print('Retried', str(num_tries + 1), 'times without success, exiting.')
                sys.exit(1)


        print('{:} logs downloaded to {:}, {:} logs skipped (already downloaded)'.format(
            n_downloaded, args.download_folder, n_skipped))


if __name__ == '__main__':
    main()
```

1. 前往 [PX4 Flight Review](https://logs.px4.io/) 下载需要分析的 `.ulg` 原始日志，并将文件放在 `px4log/px4_logs/<flight_uuid>.ulg`。
2. 运行 `step1_parse_drone_log.py` 将 `.ulg` 转成 CSV：
   - 打开脚本底部的 `log_file_name`，指向刚下载的 ULog。
   - 如需批量处理，将脚本封装成循环或使用 `python step1_parse_drone_log.py <ulog>` 这类包装命令（只要确保 `parse_ulog()` 得到正确路径）。
   - 修改 `output_file` 或重命名生成的 `drone_data_synced.csv`，保存到 `px4log/px4_logs_parsed/<flight_uuid>.csv`，这是 AESM 读取的数据接口。
3. 对历史或完整备份，可把大体量 CSV 放在 `px4log/px4_logs_parsed_all/`，仅将需要建模的子集复制/链接到 `px4log/px4_logs_parsed/`，以便 `batch_data_analysis.py` 快速扫描。

#### analyze
1. 使用 `batch_data_analysis.py` 从 `px4log/px4_logs_parsed` 读取全部 CSV，过滤掉缺少 `pressure` 或 `altitude_ellipsoid_m` 的航段，对每个航段执行：
   - 计算并对齐 GNSS 椭球高与转换后的气压高度，去除前 `BIAS_WINDOW` 样本的偏移。
   - 汇总绝对高度误差、GNSS 报告的 EPV/EPH 指标，并在成功的日志上累积统计量。
2. 结果可视化改进：
   - 直方图标题字号提高到 18，坐标轴标签提高到 15，并将 seaborn `font_scale` 设置为 1.8，以便在 `height_uncertainty_model.png` 中更容易辨识各项曲线。
   - 输出的两张直方图分别展示 `|Baro Height - GNSS Height|` 分布与 GNSS EPV 分布，便于对比模型外推与系统内自报告的精度指标。
3. 批处理完成后会生成两个核心产物：`height_uncertainty_model.png`（可直接用于报告或论文插图）与 `sensor_uncertainty_stats.json`（包括样本量、均值、标准差、拟合参数等数值基准）。

复现方式：在仓库根目录运行 `python batch_data_analysis.py`，即可重新跑完整个 AESM 管线并刷新所有统计图表。
