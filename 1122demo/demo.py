import os           #路径与文件
import json         #读写JSON
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns  #绘图

#-----------深度学习（GRU/LSTM）
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

#-----------空间插值
from pykrige.ok import OrdinaryKriging
from pykrige.rk import RegressionKriging

#-----------数据库与坐标
import psycopg2 #PostgreSQL数据库连接
from sqlalchemy import create_engine    #pandas to_sql/read_sql
from pyproj import Transformer          #坐标投影（WGS84经纬度->平面坐标）

import pickle   #保存/加载模型
import pandas as pd
from time import perf_counter
import cProfile
import pstats

#--------数据库连接配置
#DB_HOST = '10.1.3.183'
#DB_PORT = 5432
#DB_NAME = 'silas-warehouse'
#DB_USER = 'dbadmin'
#DB_PASSWORD = 'IdeaRoot@2023'

#用sqlalchemy连接数据库
#engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

engine = create_engine(
    "postgresql+psycopg2://dbadmin:IdeaRoot%402023@10.1.3.183:5432/silas-warehouse"
)

#训练/测试时间窗口
TRAIN_START = '2025-08-04 21:43:32'
TRAIN_END   = '2025-08-15 00:37:15'
TEST_START  = '2025-08-15 00:37:15'
TEST_END    = '2025-08-22 00:37:15'


#------------模型相关参数
embedding_length = 15       #时间窗口长度（暂时使用15h）
batchsize = 16              #批量大小（每个batch内包含16个时刻样本）
hidden_size = 64            #GRU隐藏维度
early_stop_loss = 0.1       #早停阈值
lr = 0.01                   #学习率
momentum = 0.9              #SGD动量

#-----------克里金参数
variogram_models = ['spherical']    #变差函数模型
n_neighbors = 8                     #RK临近点数量
n_estimators = 50                  #随机森林数量
random_state = 4                   #随机种子

#-----------投影：经纬度->平面坐标
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)    #（lng，lat）->（x，y）

def test_get_data():
    #这个代码块都是测试，等会记得删

    # 1) 看全表时间覆盖与行数
    sql_range = """
    SELECT
    COUNT(*) AS n_rows,
    MIN(processed_time) AS min_ts,
    MAX(processed_time) AS max_ts,
    current_setting('TimeZone') AS db_timezone
    FROM ods.rt_alg_env_sensor_processed;
    """
    rng = pd.read_sql(sql_range, engine)
    print("=== 表总体时间覆盖 ===")
    print(rng)

    if rng.loc[0, 'n_rows'] == 0:
        raise SystemExit("表里目前没有数据（n_rows=0），需要先确认数据是否已入库。")

    min_ts = pd.to_datetime(rng.loc[0, 'min_ts'])
    max_ts = pd.to_datetime(rng.loc[0, 'max_ts'])
    print(f"\n数据时间范围：{min_ts}  ——  {max_ts}（含两端）")

    # 2) 按天统计，快速看哪个时间段有数据
    sql_hist = """
    SELECT date_trunc('day', processed_time) AS day, COUNT(*) AS n
    FROM ods.rt_alg_env_sensor_processed
    GROUP BY 1
    ORDER BY 1;
    """
    hist = pd.read_sql(sql_hist, engine)
    print("\n=== 最近 30 天的每日条数（如果不足 30 天就全打）===")
    print(hist.tail(30))

    # 3) 自动给出一个“合理”的训练/测试窗口，并验证是否有数据
    #   优先策略A：如果跨度 >= 8 天，训练用前面的部分，测试用最后 7 天
    #   否则策略B：按 80%/20% 切分
    span_days = (max_ts - min_ts).days
    if span_days >= 8:
        TRAIN_START = min_ts
        TRAIN_END   = max_ts - pd.Timedelta(days=7)
        TEST_START  = TRAIN_END
        TEST_END    = max_ts
        strategy = "A（训练=最早到倒数第8天，测试=最后7天）"
    else:
        split_time = min_ts + (max_ts - min_ts) * 0.8
        TRAIN_START = min_ts
        TRAIN_END   = split_time
        TEST_START  = split_time
        TEST_END    = max_ts
        strategy = "B（按 80%/20% 切分）"

    print(f"\n=== 建议的切分（{strategy}）===")
    print("TRAIN_START:", TRAIN_START)
    print("TRAIN_END  :", TRAIN_END)
    print("TEST_START :", TEST_START)
    print("TEST_END   :", TEST_END)

    # 4) 验证：窗口内是否真的有数据（建议用半开区间，避免边界行重复）
    count_sql = """
    SELECT COUNT(*) AS n
    FROM ods.rt_alg_env_sensor_processed
    WHERE processed_time >= %(start)s AND processed_time < %(end)s;
    """

    cnt_train = pd.read_sql(count_sql, engine, params={'start': TRAIN_START, 'end': TRAIN_END})
    cnt_test  = pd.read_sql(count_sql, engine, params={'start': TEST_START,  'end': TEST_END})

    print("\n=== 窗口内数据量校验（半开区间）===")
    print("TRAIN 条数:", int(cnt_train.loc[0, 'n']))
    print("TEST  条数:", int(cnt_test.loc[0, 'n']))

    # 5) 若某个区间仍为 0 行，打印该区间的按日直方图，便于你微调边界
    def daily_hist(start, end, title):
        sql = """
        SELECT date_trunc('day', processed_time) AS day, COUNT(*) AS n
        FROM ods.rt_alg_env_sensor_processed
        WHERE processed_time >= %(start)s AND processed_time < %(end)s
        GROUP BY 1 ORDER BY 1;
        """
        h = pd.read_sql(sql, engine, params={'start': start, 'end': end})
        print(f"\n— {title} 每日条数 —")
        print(h.tail(30) if len(h) > 30 else h)
        return h

    if int(cnt_train.loc[0, 'n']) == 0:
        _ = daily_hist(TRAIN_START, TRAIN_END, "训练集")

    if int(cnt_test.loc[0, 'n']) == 0:
        _ = daily_hist(TEST_START, TEST_END, "测试集")


def get_data():
    # —— SQL：只写 schema.table，不要带库名
    sql = """
    SELECT
    CAST(ST_X(location) AS double precision) AS lng,
    CAST(ST_Y(location) AS double precision) AS lat,
    uid,
    original_date_str AS date,
    CAST(temperature AS double precision) AS temperature,
    CAST(altitude    AS double precision) AS altitude,
    CAST(height      AS double precision) AS height,
    CAST(humidity    AS double precision) AS humidity,
    CAST(pressure    AS double precision) AS pressure
    FROM ods.rt_alg_env_sensor_processed
    WHERE processed_time BETWEEN %(start)s AND %(end)s
    """
    # 测试
    #print(df_train.head())
    #print(df_test.head())


    # 读取
    df_train = pd.read_sql(sql, engine, params={'start': TRAIN_START, 'end': TRAIN_END})
    df_test  = pd.read_sql(sql, engine, params={'start': TEST_START,  'end': TEST_END})

    # 基础规整
    for _df in (df_train, df_test):
        _df['date'] = pd.to_datetime(_df['date'], errors='coerce')
        _df.sort_values(['uid','date'], inplace=True)
        _df.reset_index(drop=True, inplace=True)

    # —— 投影函数：写入大写 X,Y，并且把结果写回“返回值”out
    #from pyproj import Transformer
    _transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def add_projected_xy(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        X, Y = _transformer.transform(out['lng'].to_numpy(), out['lat'].to_numpy())
        out['X'] = X
        out['Y'] = Y
        return out

    df_train = add_projected_xy(df_train)   # 一定要赋回
    df_test  = add_projected_xy(df_test)


    # —— 最后再做列筛选
    cols = ['uid','date','lng','lat','X','Y','temperature','altitude','height','humidity','pressure']
    df_train = df_train[cols].copy()
    df_test  = df_test[cols].copy()

    return df_train, df_test

def prepare_data(df_train, df_test):
    #--------找出全部站点的uid，并固定顺序，便于“留一站”,注意站点必须≥3
    #uids = sorted(df_train['uid'].astype(str).unique().tolist())    #站点ID列表
    uids = sorted(pd.Index(df_train['uid'].astype(str)).union(df_test['uid'].astype(str)).tolist())
    n_stations = len(uids)

    # print(uids)
    print(n_stations)

    #-------将DataFrame转换成“分钟级齐次矩阵”
    def build_minute_matrix(df,value_col = 'temperature'):
        #以分钟为频率，按uid pivot成矩阵
        #1.先对每个站点按分钟重采样，再pivot 为列=uid， 行=分钟时间索引
        t_all_start = perf_counter()
        t0 = perf_counter()
        tmp = pd.DataFrame({
            'uid': df['uid'].astype(str),
            'date': pd.to_datetime(df['date'], errors='coerce', utc=True),
            value_col: pd.to_numeric(df[value_col], errors='coerce')
        }).dropna(subset=['uid','date', value_col])

        print(f"[build_minute_matrix] input cleaned shape={tmp.shape}, prepare={perf_counter()-t0:.3f}s")

        # 每个站点独立重采样到分钟
        from tqdm import tqdm

        t1 = perf_counter()
        pieces = []
        grouped = list(tmp.groupby('uid', sort=False))
        print(f"[build_minute_matrix] grouped stations={len(grouped)}, groupby={perf_counter()-t1:.3f}s")
        t2 = perf_counter()
        for u, g in tqdm(grouped, desc="Resampling per station"):
            s = (g.set_index('date')[value_col]
                .sort_index()
                .resample('1min').mean())

            # 站内短缺口前后填充（限制 max_gap_fill 分钟长度）
            max_gap_fill = 10

            s = s.ffill(limit=max_gap_fill).bfill(limit=max_gap_fill)
            # 时间插值（对仍有零星空洞做线性修复）
            s = s.interpolate(method='time', limit=max_gap_fill)

            # 输出为 DataFrame 便于后续 concat
            pieces.append(s.to_frame(name=u))

        if not pieces:
            # 空数据兜底（极端场景）
            return pd.DataFrame(index=pd.DatetimeIndex([], name='date'))

        # 以所有站点的并集时间轴对齐（分钟级）
        t3 = perf_counter()
        mat = pd.concat(pieces, axis=1, join='outer')
        print(f"[build_minute_matrix] concat pieces -> shape={mat.shape}, concat={perf_counter()-t3:.3f}s")

        # 统一列集合和顺序：只保留训练集合 uids
        mat.columns = mat.columns.astype(str)     # 列名与 uids 类型对齐
        cols_in_order = [str(u) for u in uids]
        if cols_in_order:                         # 只有在 uids 非空时才对齐，否则保留现有列
            t4 = perf_counter()
            mat = mat.reindex(columns=cols_in_order)
            print(f"[build_minute_matrix] reindex columns -> shape={mat.shape}, reindex={perf_counter()-t4:.3f}s")


        # 严格排序索引（填充移动到小时级别以减少计算量）
        t5 = perf_counter()
        mat = mat.sort_index()
        print(f"[build_minute_matrix] sort_index={perf_counter()-t5:.3f}s, total={perf_counter()-t_all_start:.3f}s")
        return mat


    min_train = build_minute_matrix(df_train, value_col='temperature')
    min_test  = build_minute_matrix(df_test,  value_col='temperature')

    #对齐到整点的小时聚合
    t_hour_agg = perf_counter()
    hour_train_df = min_train.resample('1H').mean()
    hour_test_df  = min_test.resample('1H').mean()
    print(f"[hourly] resample mean: {perf_counter()-t_hour_agg:.3f}s, shapes train={hour_train_df.shape}, test={hour_test_df.shape}")

    # 在小时级别进行均值填充（使用 NumPy 原地填充，通常更快）
    def _fast_fillna_with_col_means(df: pd.DataFrame) -> pd.DataFrame:
        # 确保为浮点块以便原地操作更高效
        df = df.astype(np.float32, copy=False)
        arr = df.values  # 获取底层 ndarray（单块时可原地写入）
        # 列均值（忽略 NaN）
        col_means = np.nanmean(arr, axis=0)
        # 用列均值替换 NaN
        mask = np.isnan(arr)
        if mask.any():
            row_idx, col_idx = np.where(mask)
            arr[row_idx, col_idx] = col_means[col_idx]
        return df

    t_hour_fill = perf_counter()
    hour_train_df = _fast_fillna_with_col_means(hour_train_df)
    hour_test_df  = _fast_fillna_with_col_means(hour_test_df)
    print(f"[hourly] fast fillna(mean numpy): {perf_counter()-t_hour_fill:.3f}s")

    # 若要 numpy 矩阵
    hour_train = hour_train_df.to_numpy()  # 形状: [T_train_hours, n_stations]
    #print(hour_train.shape)            #没读到数据
    hour_test  = hour_test_df.to_numpy()

    #训练集标准化
    mean_global = np.nanmean(hour_train)
    std_global  = np.nanstd(hour_train)
    std_global  = std_global if std_global > 0 else 1.0

    hour_train_z = (hour_train - mean_global) / std_global
    hour_test_z  = (hour_test  - mean_global) / std_global

    return hour_train_z, hour_test_z

if __name__ == "__main__":
    # test_get_data()
    df_train, df_test = get_data()
    # 使用 cProfile 对数据准备阶段进行性能分析
    profiler = cProfile.Profile()
    profiler.enable()
    hour_train_z, hour_test_z = prepare_data(df_train, df_test)
    profiler.disable()

    print(f"hour_train_z shape={hour_train_z.shape}, hour_test_z shape={hour_test_z.shape}")

    stats = pstats.Stats(profiler).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    print("\n=== cProfile Top (cumulative time) ===")
    stats.print_stats(30)