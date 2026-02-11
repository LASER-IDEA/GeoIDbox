try:
    # Try using the 'geoid' package (simpler, auto-downloads models)
    from geoid import GeoidHeight
    _geoid = GeoidHeight('egm96-5')  # EGM96 5-minute model
    USE_GEoid_PACKAGE = True
except ImportError:
    try:
        # Fallback to pygeodesy if geoid package not available
        from pygeodesy.ellipsoidalKarney import LatLon
        from pygeodesy import GeoidKarney
        import os
        geoid_file = "./geoids/egm2008-5.pgm"
        if os.path.exists(geoid_file):
            _geoid_interpolator = GeoidKarney(geoid_file)
            USE_GEoid_PACKAGE = False
        else:
            raise FileNotFoundError(f"Geoid file not found: {geoid_file}")
    except (ImportError, FileNotFoundError):
        raise ImportError(
            "Please install one of the following packages:\n"
            "  pip install geoid\n"
            "  or\n"
            "  pip install pygeodesy (and download geoid file)"
        )

def get_geoid_undulation(lat, lon):
    """
    计算指定经纬度的大地水准面差距 N (EGM96模型)。
    HAE = MSL + N
    """
    if USE_GEoid_PACKAGE:
        # 使用 geoid 包 (第一次运行时会自动下载约 24MB 的数据文件)
        N = _geoid(lat, lon)
    else:
        # 使用 pygeodesy
        N = _geoid_interpolator(LatLon(lat, lon))
    return N

if __name__ == "__main__":
    # 假设无人机日志中的一个位置 (例如深圳某处)
    sample_lat = 22.5431
    sample_lon = 114.0579

    N_value = get_geoid_undulation(sample_lat, sample_lon)

    print("-" * 30)
    print(f"测试位置: Lat {sample_lat}, Lon {sample_lon}")
    print(f"EGM96 大地水准面差距 (N): {N_value:.4f} 米")
    print("-" * 30)
    print("解读:")
    print(f"在该位置，如果气压计算出的MSL高度是 100米，")
    print(f"那么对应的 HAE 椭球高度大约是: 100 + ({N_value:.4f}) = {100 + N_value:.4f} 米")