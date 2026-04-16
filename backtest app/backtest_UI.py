import pandas as pd
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import pyarrow.parquet as pq
from typing import Tuple
from pathlib import Path


# ================================================================
# 自訂時間軸（顯示 HH:MM (MM.DD)）
# ================================================================
class TimeAxis(pg.AxisItem):
    TZ_MAP = {
        "New York": "America/New_York",
        "UTC": "UTC",
        "Taipei": "Asia/Taipei",
    }

    def __init__(self, *args, timezone_name: str = "New York", **kwargs):
        super().__init__(*args, **kwargs)
        self.timezone_name = timezone_name if timezone_name in self.TZ_MAP else "New York"

    def set_timezone(self, timezone_name: str):
        self.timezone_name = timezone_name if timezone_name in self.TZ_MAP else "New York"
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):
        span = max(float(maxVal) - float(minVal), 1.0)
        px = max(float(size), 1.0)
        sec_per_px = span / px
        target_step = sec_per_px * 110.0

        # Use a single tick level to avoid duplicated labels from major/minor levels.
        candidates = [300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400]
        step = candidates[-1]
        for s in candidates:
            if s >= target_step:
                step = s
                break

        start = int(np.floor(minVal / step) * step)
        end = int(np.ceil(maxVal / step) * step)
        ticks = list(range(start, end + step, step))
        return [(step, ticks)]

    def tickStrings(self, values, scale, spacing):
        labels = []
        tz_name = self.TZ_MAP.get(self.timezone_name, "America/New_York")
        for ts in values:
            try:
                dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz_name)

                if spacing >= 86400:
                    labels.append(dt.strftime("%Y.%m.%d"))
                else:
                    labels.append(f"{dt.strftime('%H:%M')}\n({dt.strftime('%Y.%m.%d')})")
            except Exception:
                labels.append("")
        return labels


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def K_bar_score(
    df: pd.DataFrame,
    BBR_thresholds: Tuple[float, float] = (0.45, 0.65),
    close_loc_thresholds: Tuple[float, float] = (0.5, 0.7),
    push_thresholds: Tuple[float, float, float] = (0.5, 1.0, 1.5),
    overlap_thresholds: Tuple[float, float] = (0.3, 0.5),
) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)}")

    bbr1, bbr2 = sorted(BBR_thresholds)
    cl1, cl2 = sorted(close_loc_thresholds)
    p1, p2, p3 = sorted(push_thresholds)
    o_tight, o_loose = sorted(overlap_thresholds)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)

    rng = high - low
    rng_safe = rng.replace(0.0, np.nan)

    body_ratio = (close - open_).abs() / rng_safe
    bull_close_loc = (close - low) / rng_safe
    bear_close_loc = (high - close) / rng_safe

    atr_14 = atr(df, 14).astype(float).replace(0.0, np.nan)
    push = (close - close.shift(1)) / atr_14

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    overlap_range = (
        pd.concat([high, prev_high], axis=1).min(axis=1)
        - pd.concat([low, prev_low], axis=1).max(axis=1)
    ).clip(lower=0.0)
    overlap_ratio = overlap_range / rng_safe

    def two_step_pts(x: pd.Series, t1: float, t2: float, mid: float, top: float) -> np.ndarray:
        xa = x.to_numpy(dtype=float)
        return np.where(xa > t2, top, np.where(xa > t1, mid, 0.0))

    def three_step_pts(x: pd.Series, t1: float, t2: float, t3: float, a: float, b: float, c: float) -> np.ndarray:
        xa = x.to_numpy(dtype=float)
        return np.where(xa > t3, c, np.where(xa > t2, b, np.where(xa > t1, a, 1.0)))

    body_pts = two_step_pts(body_ratio, bbr1, bbr2, mid=0.5, top=1.0)
    bull_close_pts = two_step_pts(bull_close_loc, cl1, cl2, mid=0.5, top=1.0)
    bull_push_pts = three_step_pts(push, p1, p2, p3, a=1.3, b=1.6, c=2.0)

    ov = overlap_ratio.to_numpy(dtype=float)
    overlap_pts = np.where(ov < o_tight, 1.0, np.where(ov < o_loose, 0.5, 0.0))

    bear_close_pts = two_step_pts(bear_close_loc, cl1, cl2, mid=0.5, top=1.0)
    bear_push_pts = three_step_pts((-push), p1, p2, p3, a=1.3, b=1.6, c=2.0)

    bull_mask = (close > open_).to_numpy()
    bear_mask = (close < open_).to_numpy()

    close_pts = np.where(bull_mask, bull_close_pts, np.where(bear_mask, bear_close_pts, 0.0))
    push_pts = np.where(bull_mask, bull_push_pts, np.where(bear_mask, bear_push_pts, 0.0))

    bull_score = (body_pts + bull_close_pts + overlap_pts) * bull_push_pts
    bear_score_abs = (body_pts + bear_close_pts + overlap_pts) * bear_push_pts
    bear_score_signed = -bear_score_abs

    score = np.zeros(len(df), dtype=float)
    score[bull_mask] = bull_score[bull_mask]
    score[bear_mask] = bear_score_signed[bear_mask]

    bull_bar_score = np.where(bull_mask, bull_score, 0.0)
    bear_bar_score = np.where(bear_mask, bear_score_signed, 0.0)

    s = pd.Series(score, index=df.index, name="bar_score").fillna(0.0)
    return pd.DataFrame(
        {
            "body_pts": body_pts,
            "close_pts": close_pts,
            "push_pts": push_pts,
            "overlap_pts": overlap_pts,
            "bull_bar_score": bull_bar_score,
            "bear_bar_score": bear_bar_score,
            "bar_score": s,
        },
        index=df.index,
    )


def K_run_score(
    df: pd.DataFrame,
    BBR_thresholds: Tuple[float, float] = (0.45, 0.65),
    close_loc_thresholds: Tuple[float, float] = (0.5, 0.7),
    push_thresholds: Tuple[float, float, float] = (0.5, 1.0, 1.5),
    overlap_thresholds: Tuple[float, float] = (0.3, 0.5),
) -> pd.DataFrame:
    kbar_df = K_bar_score(df, BBR_thresholds, close_loc_thresholds, push_thresholds, overlap_thresholds)
    bar_score = kbar_df["bar_score"]
    bull_bar_score = kbar_df["bull_bar_score"].astype(float)
    bear_bar_score = kbar_df["bear_bar_score"].astype(float)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)

    dir_ = np.sign((close - open_).to_numpy())
    dir_s = pd.Series(dir_, index=df.index)

    new_seg = (dir_s == 0) | (dir_s != dir_s.shift(1))
    seg_id = new_seg.cumsum()

    run = bar_score.where(dir_s != 0, 0.0).groupby(seg_id).cumsum()
    run = run.where(dir_s != 0, 0.0)
    run.name = "run_score"

    bull_active = dir_s == 1
    bear_active = dir_s == -1
    bull_run = bull_bar_score.where(bull_active, 0.0).groupby((~bull_active).cumsum()).cumsum().where(bull_active, 0.0)
    bear_run = bear_bar_score.where(bear_active, 0.0).groupby((~bear_active).cumsum()).cumsum().where(bear_active, 0.0)

    return pd.DataFrame(
        {
            "bar_score": bar_score,
            "seg_id": seg_id,
            "run_score": run,
            "bull_bar_score": bull_bar_score,
            "bear_bar_score": bear_bar_score,
            "bull_run": bull_run,
            "bear_run": bear_run,
        },
        index=df.index,
    )


def centered_step_edges(x: np.ndarray) -> np.ndarray:
    """Build x-edges for centered step plotting (len(edges)=len(x)+1)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)

    mids = (x[:-1] + x[1:]) * 0.5
    left_edge = x[0] - (x[1] - x[0]) * 0.5
    right_edge = x[-1] + (x[-1] - x[-2]) * 0.5
    return np.concatenate(([left_edge], mids, [right_edge]))


def centered_step_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert centered step data into plain line coordinates with equal-length x/y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    edges = centered_step_edges(x)
    step_x = np.repeat(edges, 2)[1:-1]
    step_y = np.repeat(y, 2)
    return step_x, step_y


# ================================================================
# 主程式
# ================================================================
def main():
    DATA_START_TIME = "2022-12-28 00:00"
    REPLAY_START_TIME = "2025-12-31 05:40"

    timeframe_options = ["1m", "5m", "15m", "1H", "4H", "1D"]
    product_options = ["BTC", "ETH"]

    def timeframe_to_seconds(tf: str) -> int:
        text = tf.strip()
        if text.endswith("m"):
            return int(text[:-1]) * 60
        if text.endswith("H"):
            return int(text[:-1]) * 3600
        if text.endswith("D"):
            return int(text[:-1]) * 86400
        raise ValueError(f"Unsupported timeframe: {tf}")

    coin_name = "ETH"
    timeframe = "5m"  # K 線時間週期
    market_df_cache = {}
    max_SL = 60  # 最大停損點數
    N_R_SL = 1  # 以 R 倍數計算停損
    line_mode = False
    line_start = None
    temp_line = None
    grid_lines = []
    saved_lines = []  # 保存所有劃線
    fibo_groups = []  # 保存所有斐波那契群組
    selected_item = None  # 當前選中的圖形項（線或文字）
    line_custom_pens = {}  # 存儲每條線的自訂顏色 {line_obj: pen}
    fibo_mode = False  # 斐波那契模式
    fibo_base_price = None  # 斐波那契基準價格
    text_mode = False  # 打字模式
    range_mode = False  # 價格範圍測量模式
    range_start_price = None  # 價格範圍起點
    price_ranges = []  # 保存所有價格範圍物件

    # ----------------------------- 讀取 parquet（優先 calculated） -----------------------------
    data_dir = Path("backtest app/data source")

    def build_calculated_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
        work_df = raw_df.copy()
        work_df["dt_utc"] = pd.to_datetime(work_df["dt_utc"], utc=True)
        work_df["dt_ny"] = work_df["dt_utc"].dt.tz_convert("America/New_York")
        work_df["dt_tp"] = work_df["dt_utc"].dt.tz_convert("Asia/Taipei")
        work_df["dt_based"] = work_df["dt_ny"]
        work_df["dt_based_ts"] = work_df["dt_based"].map(lambda x: x.timestamp())
        work_df["date"] = work_df["dt_ny"].dt.date
        work_df = work_df.sort_values(["date", "dt_ny"]).reset_index(drop=True)
        work_df["bar_index"] = work_df.groupby("date").cumcount() + 1
        work_df["MA20"] = work_df["close"].ewm(span=20, adjust=False).mean()
        work_df["MA100"] = work_df["close"].ewm(span=100, adjust=False).mean()

        # 價格精度：以當日第一根開盤價判斷，<10 用 4 位，>=10 用 2 位
        work_df["day_open_price"] = work_df.groupby("date")["open"].transform("first")
        work_df["price_precision"] = work_df["day_open_price"].apply(lambda v: 4 if v < 10 else 2).astype(int)

        # ================================================================
        # ⭐ 新增 Body Ratio / ΔC 計算在 df
        # ================================================================
        work_df["body_ratio"] = (work_df["close"] - work_df["open"]).abs() / (work_df["high"] - work_df["low"])
        work_df["body_ratio"] = 100 * work_df["body_ratio"].replace([float('inf'), -float('inf')], 0).fillna(0)
        work_df["body_ratio_level"] = pd.cut(work_df["body_ratio"], bins=[-1, 34, 68, 101], labels=['L', 'M', 'H'])
        work_df["K_range"] = (work_df["high"] - work_df["low"]) * 100 / work_df["open"]
        work_df["K_range_level"] = pd.cut(work_df["K_range"], bins=[-1, 0.35, 0.6, 100], labels=['S', 'M', 'L'])

        # K1 C2_profit_R calculation
        # if K0 is bear bar C2_profit_R = (K0_low - K1_close)/K0_K_range
        # if K0 bull bar C2_profit_R = (K1_close - K0_high)/K0_K_range
        work_df["C2_profit_R"] = 0.0
        work_df["Max_K2_profit_R"] = 0.0
        work_df["0.4R"] = 0.0
        work_df["0.7R"] = 0.0
        work_df["1.0R"] = 0.0
        work_df["1.3R"] = 0.0
        work_df["1.6R"] = 0.0
        work_df["2.0R"] = 0.0
        work_df["SL_price"] = 0.0
        work_df["entry_price"] = 0.0
        work_df["size"] = 0.0
        work_df["shadow_ratio"] = 0.0
        work_df["counter_shadow_ratio"] = 0.0
        # for i in range(len(work_df)):
    #     if i != 0:
    #         if df.at[i-1, "close"] < df.at[i-1, "open"]:  # K0 is bear bar
    #             df.at[i, "C2_profit_R"] = (df.at[i-1, "low"] - df.at[i, "close"]) / ((df.at[i-1, "high"] - df.at[i-1, "low"]) * N_R_SL)
    #             df.at[i, "Max_K2_profit_R"] = (df.at[i-1, "low"] - df.at[i, "low"]) / ((df.at[i-1, "high"] - df.at[i-1, "low"]) * N_R_SL)
    #         else:  # K0 is bull bar
    #             df.at[i, "C2_profit_R"] = (df.at[i, "close"] - df.at[i-1, "high"]) / ((df.at[i-1, "high"] - df.at[i-1, "low"]) * N_R_SL)
    #             df.at[i, "Max_K2_profit_R"] = (df.at[i, "high"] - df.at[i-1, "high"]) / ((df.at[i-1, "high"] - df.at[i-1, "low"]) * N_R_SL)
    #     if df.at[i, "close"] < df.at[i, "open"]:  # K0 is bear bar
    #         df.at[i, "0.4R"] = df.at[i, "low"] - 0.4 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "0.7R"] = df.at[i, "low"] - 0.7 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "1.0R"] = df.at[i, "low"] - 1.0 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "1.3R"] = df.at[i, "low"] - 1.3 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "1.6R"] = df.at[i, "low"] - 1.6 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "2.0R"] = df.at[i, "low"] - 2.0 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "SL_price"] = df.at[i, "low"] + N_R_SL * (df.at[i, "high"] - df.at[i, "low"])
    #         df.at[i, "entry_price"] = df.at[i, "low"]
    #         df.at[i, "size"] = max_SL / (N_R_SL * (df.at[i, "high"] - df.at[i, "low"]))
    #         df.at[i, "shadow_ratio"] = (df.at[i, "high"] - df.at[i, "open"])*100/(df.at[i, "high"] - df.at[i, "low"])
    #         df.at[i, "counter_shadow_ratio"] = (df.at[i, "close"] - df.at[i, "low"])*100/(df.at[i, "high"] - df.at[i, "low"])
            
    #     else:  # K0 is bull bar
    #         df.at[i, "0.4R"] = df.at[i, "high"] + 0.4 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "0.7R"] = df.at[i, "high"] + 0.7 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "1.0R"] = df.at[i, "high"] + 1.0 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "1.3R"] = df.at[i, "high"] + 1.3 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "1.6R"] = df.at[i, "high"] + 1.6 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "2.0R"] = df.at[i, "high"] + 2.0 * (df.at[i, "high"] - df.at[i, "low"]) * N_R_SL
    #         df.at[i, "SL_price"] = df.at[i, "high"] - N_R_SL * (df.at[i, "high"] - df.at[i, "low"])
    #         df.at[i, "entry_price"] = df.at[i, "high"]
    #         df.at[i, "size"] = max_SL / (N_R_SL * (df.at[i, "high"] - df.at[i, "low"]))
    #         df.at[i, "shadow_ratio"] = (df.at[i, "open"] - df.at[i, "low"])*100/(df.at[i, "high"] - df.at[i, "low"])
    #         df.at[i, "counter_shadow_ratio"] = (df.at[i, "high"] - df.at[i, "close"])*100/(df.at[i, "high"] - df.at[i, "low"])
        
    

        work_df["delta_c"] = work_df["close"].diff().fillna(0) * 100 / work_df["close"]

        k_bar_score_df_local = K_bar_score(work_df)
        k_run_score_df_local = K_run_score(work_df)
        work_df["kbar_score"] = k_bar_score_df_local["bar_score"].astype(float)
        work_df["run_score"] = k_run_score_df_local["run_score"].astype(float)
        work_df["bull_bar_score"] = k_run_score_df_local["bull_bar_score"].astype(float)
        work_df["bear_bar_score"] = k_run_score_df_local["bear_bar_score"].astype(float)
        work_df["bull_run"] = k_run_score_df_local["bull_run"].astype(float)
        work_df["bear_run"] = k_run_score_df_local["bear_run"].astype(float)
        return work_df

    required_calculated_cols = {
        "dt_utc", "dt_ny", "dt_tp", "MA20", "MA100", "price_precision",
        "body_ratio", "K_range", "delta_c", "kbar_score", "run_score",
        "bull_bar_score", "bear_bar_score", "bull_run", "bear_run"
    }

    def load_market_dataframe(product_name: str, timeframe_value: str) -> pd.DataFrame:
        cache_key = (product_name, timeframe_value)
        if cache_key in market_df_cache:
            print(f"Loaded dataframe from memory cache: {product_name}_{timeframe_value}")
            return market_df_cache[cache_key].copy(deep=True)

        raw_path = data_dir / f"{product_name}_{timeframe_value}.parquet"
        calculated_path = data_dir / f"{product_name}_{timeframe_value}_calculated.parquet"

        loaded_df = None
        if calculated_path.exists():
            try:
                cached_df = pq.read_table(str(calculated_path)).to_pandas()
                if required_calculated_cols.issubset(set(cached_df.columns)):
                    loaded_df = cached_df
                    print(f"Loaded calculated parquet: {calculated_path.name}")
                else:
                    print(f"Calculated parquet missing required columns, rebuilding: {calculated_path.name}")
            except Exception as e:
                print(f"Failed to read calculated parquet ({calculated_path.name}), rebuilding from raw. reason={e}")

        if loaded_df is None:
            raw_df = pq.read_table(str(raw_path)).to_pandas()
            loaded_df = build_calculated_dataframe(raw_df)
            try:
                loaded_df.to_parquet(calculated_path, index=False)
                print(f"Saved calculated parquet: {calculated_path.name}")
            except Exception as e:
                print(f"Failed to save calculated parquet ({calculated_path.name}). reason={e}")

        # 第一個時間：控制可用資料最早起點
        data_start_dt = pd.to_datetime(DATA_START_TIME).tz_localize("America/New_York")
        loaded_df = loaded_df[loaded_df["dt_ny"] >= data_start_dt]
        loaded_df = loaded_df.reset_index(drop=True)

        if len(loaded_df) == 0:
            raise ValueError(f"DATA_START_TIME {DATA_START_TIME} 超出資料範圍")

        loaded_df["bar_index"] = loaded_df.index
        market_df_cache[cache_key] = loaded_df
        return loaded_df.copy(deep=True)

    df = load_market_dataframe(coin_name, timeframe)

    price_precision_default = int(df["price_precision"].iloc[0]) if len(df) > 0 else 2
    time_base_name = "New York"

    time_col_by_name = {
        "New York": "dt_ny",
        "UTC": "dt_utc",
        "Taipei": "dt_tp",
    }

    def set_time_base(name: str):
        nonlocal time_base_name
        chosen = name if name in time_col_by_name else "New York"
        time_base_name = chosen
        base_col = time_col_by_name[chosen]
        df["dt_based"] = df[base_col]
        # Vectorized conversion avoids per-row Python calls.
        df["dt_based_ts"] = df["dt_based"].astype("int64").to_numpy(dtype=np.float64) / 1_000_000_000.0

    def get_replay_start_index(df_obj: pd.DataFrame) -> int:
        replay_start_dt_ny_local = pd.to_datetime(REPLAY_START_TIME).tz_localize("America/New_York")
        replay_idx = df_obj[df_obj["dt_ny"] >= replay_start_dt_ny_local].index.min()
        if pd.isna(replay_idx):
            raise ValueError(f"REPLAY_START_TIME {REPLAY_START_TIME} 超出資料範圍")
        return int(replay_idx)

    set_time_base(time_base_name)

    replay_start_idx = get_replay_start_index(df)

    idx = replay_start_idx
    visible_first_dt_ny = None
    visible_last_dt_ny = None


    # ================================================================
    # 建立 QApplication
    # ================================================================
    app = QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Calibri", 12, QtGui.QFont.Bold))

    # ================================================================
    # 建立 UI
    # ================================================================
    # 頂層視窗，左側為圖表，右側為下單面板
    root = QtWidgets.QWidget()
    root.setWindowTitle(f"{coin_name} Replay UI {timeframe}")
    hbox = QtWidgets.QHBoxLayout(root)
    hbox.setContentsMargins(6, 6, 6, 6)
    hbox.setSpacing(6)

    # 左側容器（工具栏 + 圖表）
    left_container = QtWidgets.QWidget()
    left_layout = QtWidgets.QVBoxLayout(left_container)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(2)

    # 工具栏
    toolbar = QtWidgets.QWidget()
    toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
    toolbar_layout.setContentsMargins(2, 2, 2, 2)
    toolbar_layout.setSpacing(5)

    btn_h_line = QtWidgets.QPushButton("H")
    btn_h_line.setFixedSize(40, 30)
    btn_h_line.setToolTip("畫水平線")
    
    btn_l_line = QtWidgets.QPushButton("L")
    btn_l_line.setFixedSize(40, 30)
    btn_l_line.setToolTip("畫普通線")

    btn_fibo = QtWidgets.QPushButton("Fibo")
    btn_fibo.setFixedSize(50, 30)
    btn_fibo.setToolTip("斐波那契水平線 (倍數: 0, 0.5, 1, 2, 3)")
    
    btn_text = QtWidgets.QPushButton("T")
    btn_text.setFixedSize(40, 30)
    btn_text.setToolTip("打字")

    btn_range = QtWidgets.QPushButton("📏")
    btn_range.setFixedSize(40, 30)
    btn_range.setToolTip("測量價格範圍")

    btn_screenshot = QtWidgets.QPushButton("Shot")
    btn_screenshot.setFixedSize(60, 30)
    btn_screenshot.setToolTip("截圖並匯出圖表")

    btn_auto_scale = QtWidgets.QPushButton("Auto")
    btn_auto_scale.setFixedSize(60, 30)
    btn_auto_scale.setToolTip("以目前視窗內資料自動縮放")

    btn_auto_all = QtWidgets.QPushButton("AutoAll")
    btn_auto_all.setFixedSize(70, 30)
    btn_auto_all.setToolTip("以全部 index 資料自動縮放")

    jump_date_edit = QtWidgets.QDateEdit()
    jump_date_edit.setCalendarPopup(True)
    jump_date_edit.setDisplayFormat("yyyy-MM-dd")
    jump_date_edit.setFixedWidth(120)
    jump_date_edit.setStyleSheet(
        "background-color:#0d1420; border:1px solid #4d5a73; border-radius:3px; color:#e6edf7; padding:2px 6px;"
    )

    jump_time_edit = QtWidgets.QTimeEdit()
    jump_time_edit.setDisplayFormat("HH:mm")
    jump_time_edit.setFixedWidth(90)
    jump_time_edit.setStyleSheet(
        "background-color:#0d1420; border:1px solid #4d5a73; border-radius:3px; color:#e6edf7; padding:2px 6px;"
    )

    btn_run_from_dt = QtWidgets.QPushButton("Run")
    btn_run_from_dt.setFixedSize(60, 30)
    btn_run_from_dt.setToolTip("以指定日期時間重設 replay 起點並重繪")

    toolbar_layout.addWidget(btn_h_line)
    toolbar_layout.addWidget(btn_l_line)
    toolbar_layout.addWidget(btn_fibo)
    toolbar_layout.addWidget(btn_text)
    toolbar_layout.addWidget(btn_range)
    toolbar_layout.addWidget(btn_screenshot)
    toolbar_layout.addWidget(btn_auto_scale)
    toolbar_layout.addWidget(btn_auto_all)
    toolbar_layout.addWidget(jump_date_edit)
    toolbar_layout.addWidget(jump_time_edit)
    toolbar_layout.addWidget(btn_run_from_dt)
    toolbar_layout.addStretch()

    left_layout.addWidget(toolbar)

    # 固定 K 線資訊欄（工具列下方、圖表上方）
    kline_info_panel = QtWidgets.QWidget()
    kline_info_panel.setStyleSheet("background-color: #121722; border: 1px solid #2a3142; border-radius: 4px;")
    kline_info_layout = QtWidgets.QVBoxLayout(kline_info_panel)
    kline_info_layout.setContentsMargins(10, 6, 10, 6)
    kline_info_layout.setSpacing(2)

    market_selector_row = QtWidgets.QHBoxLayout()
    market_selector_row.setContentsMargins(0, 0, 0, 0)
    market_selector_row.setSpacing(8)

    product_combo = QtWidgets.QComboBox()
    product_combo.addItems(product_options)
    product_combo.setCurrentText(coin_name)
    product_combo.setFixedWidth(100)
    product_combo.setStyleSheet(
        "QComboBox {"
        "background-color:#0d1420; border:1px solid #6b7fa1; border-radius:2px; "
        "color:#f5f7fa; padding:2px 6px; font-size:12pt; font-weight:700;"
        "}"
        "QComboBox QAbstractItemView {"
        "background-color:#0d1420; color:#e6edf7; selection-background-color:#2a3952;"
        "}"
    )

    timeframe_combo = QtWidgets.QComboBox()
    timeframe_combo.addItems(timeframe_options)
    timeframe_combo.setCurrentText(timeframe)
    timeframe_combo.setFixedWidth(100)
    timeframe_combo.setStyleSheet(
        "QComboBox {"
        "background-color:#0d1420; border:1px solid #6b7fa1; border-radius:2px; "
        "color:#f5f7fa; padding:2px 6px; font-size:12pt; font-weight:700;"
        "}"
        "QComboBox QAbstractItemView {"
        "background-color:#0d1420; color:#e6edf7; selection-background-color:#2a3952;"
        "}"
    )

    timeframe_loading_label = QtWidgets.QLabel("loading...")
    timeframe_loading_label.setStyleSheet("color:#ffcc80; font-size:10pt; font-weight:700;")
    timeframe_loading_label.setVisible(False)

    market_selector_row.addWidget(product_combo, 0)
    market_selector_row.addWidget(timeframe_combo, 0)
    market_selector_row.addWidget(timeframe_loading_label, 0)
    market_selector_row.addStretch(1)

    kline_ohlc_label = QtWidgets.QLabel("開=--  高=--  低=--  收=--  +0.00(+0.00%)")
    kline_ohlc_label.setStyleSheet("color: #cfd7e6; font-size: 12pt; font-weight: 700;")

    kline_ema_label = QtWidgets.QLabel("EMA 20=--")
    kline_ema_label.setStyleSheet("color: #89a8ff; font-size: 12pt; font-weight: 700;")

    kline_info_layout.addLayout(market_selector_row)
    kline_info_layout.addWidget(kline_ohlc_label)
    kline_info_layout.addWidget(kline_ema_label)
    left_layout.addWidget(kline_info_panel)

    win = pg.GraphicsLayoutWidget(title=f"{coin_name} Replay UI {timeframe}")
    left_layout.addWidget(win, 4)

    # 固定指標資訊欄（樣式與價格資訊欄一致）
    indicator_info_panel = QtWidgets.QWidget()
    indicator_info_panel.setStyleSheet("background-color: #121722; border: 1px solid #2a3142; border-radius: 4px;")
    indicator_info_layout = QtWidgets.QVBoxLayout(indicator_info_panel)
    indicator_info_layout.setContentsMargins(10, 6, 10, 6)
    indicator_info_layout.setSpacing(2)

    indicator_info_row = QtWidgets.QHBoxLayout()
    indicator_info_row.setContentsMargins(0, 0, 0, 0)
    indicator_info_row.setSpacing(8)

    indicator_info_label = QtWidgets.QLabel(
        "<span style='font-size:12pt; color:#cfd7e6;'>SBS Run Score&nbsp;&nbsp;</span>"
        "<span style='font-size:12pt; color:#ffcc80;'>bull bar=&nbsp;#.#</span>"
        "<span style='font-size:12pt; color:#7f8897;'>&nbsp;&nbsp;</span>"
        "<span style='font-size:12pt; color:#089981;'>bull run=&nbsp;#.#</span>"
        "<span style='font-size:12pt; color:#7f8897;'>&nbsp;&nbsp;</span>"
        "<span style='font-size:12pt; color:#f48fb1;'>bear bar=&nbsp;#.#</span>"
        "<span style='font-size:12pt; color:#7f8897;'>&nbsp;&nbsp;</span>"
        "<span style='font-size:12pt; color:#e53935;'>bear run=&nbsp;#.#</span>"
    )
    indicator_info_label.setStyleSheet("color: #cfd7e6; font-size: 12pt; font-weight: 700;")
    indicator_info_label.setTextFormat(QtCore.Qt.RichText)

    indicator_x_value_badge = QtWidgets.QLabel("--:-- (----.--.--)")
    indicator_x_value_badge.setStyleSheet(
        "background-color:#0d1420; border:1px solid #6b7fa1; border-radius:2px; "
        "color:#e6edf7; padding:2px 6px; font-size:11pt;"
    )
    indicator_x_value_badge.setAlignment(QtCore.Qt.AlignCenter)

    timezone_combo = QtWidgets.QComboBox()
    timezone_combo.addItems(["New York", "UTC", "Taipei"])
    timezone_combo.setCurrentText(time_base_name)
    timezone_combo.setFixedWidth(110)
    timezone_combo.setStyleSheet(
        "QComboBox {"
        "background-color:#0d1420; border:1px solid #6b7fa1; border-radius:2px; "
        "color:#e6edf7; padding:2px 6px; font-size:11pt;"
        "}"
        "QComboBox QAbstractItemView {"
        "background-color:#0d1420; color:#e6edf7; selection-background-color:#2a3952;"
        "}"
    )

    indicator_info_row.addWidget(indicator_info_label, 1)
    indicator_info_row.addWidget(timezone_combo, 0)
    indicator_info_row.addWidget(indicator_x_value_badge, 0)
    indicator_info_layout.addLayout(indicator_info_row)
    left_layout.addWidget(indicator_info_panel)

    indicator_win = pg.GraphicsLayoutWidget()
    left_layout.addWidget(indicator_win, 2)

    hbox.addWidget(left_container, 3)

    time_axis = TimeAxis(orientation="bottom", timezone_name=time_base_name)
    pg.setConfigOption('background', '#181c27')
    pg.setConfigOption('foreground', 'white')

    # ================================================================
    # 上方 K 線窗格
    # ================================================================
    plot = win.addPlot(row=0, col=0)
    plot.getViewBox().setBackgroundColor("#181c27")

    plot.setAutoVisible(y=True)
    plot.showGrid(x=False, y=True, alpha=0.3)
    plot.getAxis("right").setStyle(maxTickLevel=0)
    plot.getAxis("bottom").setStyle(maxTickLevel=0)

    plot.showAxis("right", True)
    plot.showAxis("left", False)
    plot.setLabel("right", "Price")
    plot.showAxis("bottom", False)
    plot.hideButtons()

    # ================================================================
    # 下方技術指標窗格（X 軸與 K 線對齊）
    # ================================================================
    indicator_plot = indicator_win.addPlot(axisItems={"bottom": time_axis}, row=0, col=0)
    indicator_plot.getViewBox().setBackgroundColor("#141822")
    indicator_plot.showGrid(x=False, y=True, alpha=0.2)
    indicator_plot.getAxis("right").setStyle(maxTickLevel=0)
    indicator_plot.getAxis("bottom").setStyle(maxTickLevel=0)
    indicator_plot.showAxis("left", False)
    indicator_plot.showAxis("right", True)
    indicator_plot.setLabel("right", "Score")
    indicator_plot.setYRange(-15, 15, padding=0)
    indicator_plot.enableAutoRange(y=False)
    indicator_plot.hideButtons()

    # 固定左右窗格右軸寬度，避免不同位數導致繪圖區域錯位
    right_axis_width = 72
    plot.getAxis("right").setWidth(right_axis_width)
    indicator_plot.getAxis("right").setWidth(right_axis_width)

    indicator_plot.setXLink(plot)

    # ================================================================
    # 十字游標線
    # ================================================================
    vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("gray", width=1))
    hline = pg.InfiniteLine(angle=0, pen=pg.mkPen("gray", width=1))
    vline_indicator = pg.InfiniteLine(angle=90, pen=pg.mkPen("gray", width=1))
    hline_indicator = pg.InfiniteLine(angle=0, pen=pg.mkPen("gray", width=1))
    plot.addItem(vline, ignoreBounds=True)
    plot.addItem(hline, ignoreBounds=True)
    indicator_plot.addItem(vline_indicator, ignoreBounds=True)
    indicator_plot.addItem(hline_indicator, ignoreBounds=True)
    current_crosshair_active = "none"

    def quantize_step(val: float, step: float) -> float:
        return round(val / step) * step

    def set_crosshair_visibility(active_name: str):
        if active_name == "main":
            vline.show()
            vline_indicator.show()
            hline.show()
            hline_indicator.hide()
        elif active_name == "indicator":
            vline.show()
            vline_indicator.show()
            hline.hide()
            hline_indicator.show()
        else:
            vline.hide()
            vline_indicator.hide()
            hline.hide()
            hline_indicator.hide()

    def cursor_in_main_plot() -> bool:
        p = win.mapFromGlobal(QtGui.QCursor.pos())
        scene_p = win.mapToScene(p)
        return plot.vb.sceneBoundingRect().contains(scene_p)

    def cursor_in_indicator_plot() -> bool:
        p = indicator_win.mapFromGlobal(QtGui.QCursor.pos())
        scene_p = indicator_win.mapToScene(p)
        return indicator_plot.vb.sceneBoundingRect().contains(scene_p)

    def refresh_crosshair_state_by_cursor():
        nonlocal current_crosshair_active
        in_main = cursor_in_main_plot()
        in_indicator = cursor_in_indicator_plot()

        if in_main and not in_indicator:
            desired = "main"
        elif in_indicator and not in_main:
            desired = "indicator"
        else:
            desired = "none"

        if desired != current_crosshair_active:
            current_crosshair_active = desired
            set_crosshair_visibility(desired)

    crosshair_state_timer = QtCore.QTimer(root)
    crosshair_state_timer.setInterval(40)
    crosshair_state_timer.timeout.connect(refresh_crosshair_state_by_cursor)
    crosshair_state_timer.start()

    set_crosshair_visibility("none")

    # 軸上數值方框（最上層）
    axis_value_style = "background-color:#0d1420; border:1px solid #6b7fa1; border-radius:2px; color:#e6edf7; padding:2px 6px;"

    price_axis_value_label = QtWidgets.QLabel(win)
    price_axis_value_label.setStyleSheet(axis_value_style)
    price_axis_value_label.setAlignment(QtCore.Qt.AlignCenter)
    price_axis_value_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
    price_axis_value_label.raise_()

    indicator_axis_value_label = QtWidgets.QLabel(indicator_win)
    indicator_axis_value_label.setStyleSheet(axis_value_style)
    indicator_axis_value_label.setAlignment(QtCore.Qt.AlignCenter)
    indicator_axis_value_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
    indicator_axis_value_label.raise_()

    # Y軸價格方形左側的 [S] 和 [L] 按鈕
    button_style_s = "background-color:#c41e3a; border:1px solid #8b0000; border-radius:2px; color:#ffffff; padding:2px 4px; font-weight:bold;"
    button_style_l = "background-color:#1e5a96; border:1px solid #0d3d73; border-radius:2px; color:#ffffff; padding:2px 4px; font-weight:bold;"

    btn_short = QtWidgets.QPushButton("S", win)
    btn_short.setFixedSize(24, 24)
    btn_short.setStyleSheet(button_style_s)
    btn_short.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
    btn_short.raise_()
    btn_short.setToolTip("Short (賣出信號)")

    btn_long = QtWidgets.QPushButton("L", win)
    btn_long.setFixedSize(24, 24)
    btn_long.setStyleSheet(button_style_l)
    btn_long.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
    btn_long.raise_()
    btn_long.setToolTip("Long (買入信號)")

    # ================================================================
    # 按鈕功能連接
    # ================================================================
    def on_h_line_clicked():
        nonlocal line_mode
        line_mode = "horizontal"

    def on_l_line_clicked():
        nonlocal line_mode, line_start, temp_line
        line_mode = True
        line_start = None
        if temp_line:
            plot.removeItem(temp_line)
            temp_line = None

    def on_fibo_clicked():
        nonlocal line_mode, fibo_mode, fibo_base_price
        fibo_mode = True
        fibo_base_price = None
        fibo_manager.start_mode()

    def on_text_clicked():
        nonlocal text_mode
        text_mode = True

    def on_range_clicked():
        nonlocal range_mode, range_start_price
        range_mode = True
        range_start_price = None

    def on_screenshot_clicked():
        pixmap = left_container.grab()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "匯出截圖",
            "chart.png",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
        )
        if not file_path:
            return
        ok = pixmap.save(file_path)
        if ok:
            QtWidgets.QMessageBox.information(None, "截圖完成", f"已儲存:\n{file_path}")
        else:
            QtWidgets.QMessageBox.critical(None, "截圖失敗", "無法儲存檔案")

    btn_h_line.clicked.connect(on_h_line_clicked)
    btn_l_line.clicked.connect(on_l_line_clicked)
    btn_fibo.clicked.connect(on_fibo_clicked)
    btn_text.clicked.connect(on_text_clicked)
    btn_range.clicked.connect(on_range_clicked)
    btn_screenshot.clicked.connect(on_screenshot_clicked)

    auto_all_mode = False
    syncing_auto_all_range = False

    def sync_jump_datetime_controls(row_idx: int | None = None):
        if len(df) == 0:
            return
        default_idx = replay_start_idx if idx <= replay_start_idx else (idx - 1)
        row_idx = min(max(default_idx if row_idx is None else row_idx, 0), len(df) - 1)
        dt_min = df.iloc[0]["dt_based"]
        dt_max = df.iloc[len(df) - 1]["dt_based"]
        dt_cur = df.iloc[row_idx]["dt_based"]
        jump_date_edit.setMinimumDate(QtCore.QDate(dt_min.year, dt_min.month, dt_min.day))
        jump_date_edit.setMaximumDate(QtCore.QDate(dt_max.year, dt_max.month, dt_max.day))
        jump_date_edit.setDate(QtCore.QDate(dt_cur.year, dt_cur.month, dt_cur.day))
        jump_time_edit.setTime(QtCore.QTime(dt_cur.hour, dt_cur.minute))

    def get_current_visible_candle_bounds_ny() -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if len(df) == 0:
            return None

        appeared_end = min(max(idx, replay_start_idx), len(df))
        if appeared_end <= replay_start_idx:
            return None

        appeared = df.iloc[replay_start_idx:appeared_end]
        x0, x1 = plot.viewRange()[0]
        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        visible = appeared[(appeared["dt_based_ts"] >= x_min) & (appeared["dt_based_ts"] <= x_max)]
        if len(visible) == 0:
            return None

        first_dt = pd.Timestamp(visible.iloc[0]["dt_ny"])
        last_dt = pd.Timestamp(visible.iloc[-1]["dt_ny"])
        if last_dt < first_dt:
            first_dt, last_dt = last_dt, first_dt
        return first_dt, last_dt

    def refresh_visible_candle_record():
        nonlocal visible_first_dt_ny, visible_last_dt_ny
        bounds = get_current_visible_candle_bounds_ny()
        if bounds is None:
            return
        visible_first_dt_ny, visible_last_dt_ny = bounds

    def apply_auto_all_view():
        nonlocal syncing_auto_all_range
        if len(df) == 0:
            return

        appeared_end = min(max(idx, replay_start_idx), len(df))
        appeared = df.iloc[replay_start_idx:appeared_end]
        if len(appeared) == 0:
            return

        one_tick_sec = float(timeframe_to_seconds(timeframe))
        x_min = float(appeared["dt_based_ts"].min()) - one_tick_sec
        x_max = float(appeared["dt_based_ts"].max()) + one_tick_sec
        if x_max <= x_min:
            x_max = x_min + 1.0

        low_min = float(appeared["low"].min())
        high_max = float(appeared["high"].max())
        span = high_max - low_min
        pad = max(span * 0.08, max(abs(high_max), abs(low_min), 1.0) * 0.002)

        bull_vals = appeared["bull_run"].to_numpy(dtype=float)
        bear_vals = appeared["bear_run"].to_numpy(dtype=float)
        bull_finite = bull_vals[np.isfinite(bull_vals)]
        bear_finite = bear_vals[np.isfinite(bear_vals)]
        bull_max = float(np.max(bull_finite)) if bull_finite.size > 0 else 0.0
        bear_min = float(np.min(bear_finite)) if bear_finite.size > 0 else 0.0

        syncing_auto_all_range = True
        try:
            plot.setXRange(x_min, x_max, padding=0.01)
            plot.setYRange(low_min - pad, high_max + pad, padding=0)
            indicator_plot.setYRange(bear_min - 3.0, bull_max + 3.0, padding=0.03)
        finally:
            syncing_auto_all_range = False

        update_indicator_value_panel(appeared_end - 1)

    def on_auto_scale_clicked():
        nonlocal auto_all_mode
        if len(df) == 0:
            return
        auto_all_mode = False
        anchor_idx = min(max(idx - 1, replay_start_idx), len(df) - 1)
        x0, x1 = plot.viewRange()[0]
        one_tick_sec = float(timeframe_to_seconds(timeframe))
        if x1 <= x0:
            x1 = x0 + one_tick_sec
        plot.setXRange(x0 - one_tick_sec, x1 + one_tick_sec, padding=0)
        update_main_axis_range(anchor_idx)
        update_indicator_axis_range(anchor_idx)
        update_indicator_value_panel(anchor_idx)

    btn_auto_scale.clicked.connect(on_auto_scale_clicked)

    def on_auto_all_clicked():
        nonlocal auto_all_mode
        if len(df) == 0:
            return
        auto_all_mode = True
        apply_auto_all_view()

    btn_auto_all.clicked.connect(on_auto_all_clicked)

    def on_run_from_datetime_clicked():
        nonlocal idx, replay_start_idx, position, realized_pnl, realized_r_pnl, order_id_seq
        nonlocal line_mode, line_start, temp_line, selected_item, fibo_mode, fibo_base_price, text_mode, range_mode, range_start_price
        if len(df) == 0:
            return

        tz_name = TimeAxis.TZ_MAP.get(time_base_name, "America/New_York")
        date_txt = jump_date_edit.date().toString("yyyy-MM-dd")
        time_txt = jump_time_edit.time().toString("HH:mm")
        try:
            target_dt = pd.Timestamp(f"{date_txt} {time_txt}").tz_localize(tz_name)
        except Exception:
            QtWidgets.QMessageBox.warning(None, "時間格式錯誤", "無法解析指定的日期時間")
            return

        target_idx = df[df["dt_based"] >= target_dt].index.min()
        if pd.isna(target_idx):
            QtWidgets.QMessageBox.warning(None, "超出範圍", "指定時間超出目前資料範圍")
            return

        replay_start_idx = int(target_idx)
        idx = replay_start_idx

        # 清空 replay 交易/畫線/標記資料
        position = None
        realized_pnl = 0.0
        realized_r_pnl = 0.0
        pending_orders.clear()
        trade_markers.clear()
        saved_lines.clear()
        fibo_groups.clear()
        line_custom_pens.clear()
        price_ranges.clear()
        line_tool_manager.clear_all()
        order_id_seq = 1
        line_mode = False
        line_start = None
        temp_line = None
        selected_item = None
        fibo_mode = False
        fibo_base_price = None
        fibo_manager.cancel_mode()
        text_mode = False
        range_mode = False
        range_start_price = None

        refresh_orders_table()
        refresh_trades_table()
        redraw_all()
        update_pnl_labels()

        anchor_idx = min(max(replay_start_idx, 0), len(df) - 1)
        update_indicator_value_panel(anchor_idx)
        update_kline_info_panel(anchor_idx)
        update_axis_value_boxes(anchor_idx)
        sync_jump_datetime_controls(anchor_idx)

    btn_run_from_dt.clicked.connect(on_run_from_datetime_clicked)

    def on_timezone_changed(chosen_name: str):
        if len(df) == 0:
            return

        x0, x1 = plot.viewRange()[0]
        set_time_base(chosen_name)
        time_axis.set_timezone(time_base_name)

        redraw_all()
        plot.setXRange(x0, x1, padding=0)

        anchor_idx = min(max(idx - 1, replay_start_idx), len(df) - 1)
        update_indicator_value_panel(anchor_idx)
        update_kline_info_panel(anchor_idx)
        update_axis_value_boxes(anchor_idx)
        sync_jump_datetime_controls(anchor_idx)

    timezone_combo.currentTextChanged.connect(on_timezone_changed)
    default_replay_dt = pd.to_datetime(REPLAY_START_TIME).tz_localize("America/New_York").tz_convert(
        TimeAxis.TZ_MAP.get(time_base_name, "America/New_York")
    )
    jump_date_edit.setDate(QtCore.QDate(default_replay_dt.year, default_replay_dt.month, default_replay_dt.day))
    jump_time_edit.setTime(QtCore.QTime(default_replay_dt.hour, default_replay_dt.minute))
    sync_jump_datetime_controls(replay_start_idx)

    # ================================================================
    # ⭐⭐ 模擬下單面板（多空、類型、價格、PnL）
    # 使用 LayoutWidget 內嵌標準 Qt 控制項
    # ================================================================
    order_panel = pg.LayoutWidget()
    order_panel.setMinimumWidth(260)
    hbox.addWidget(order_panel, 1)

    lbl_side = QtWidgets.QLabel("Side")
    side_combo = QtWidgets.QComboBox()
    side_combo.addItems(["long", "short"])

    lbl_type = QtWidgets.QLabel("Type")
    type_combo = QtWidgets.QComboBox()
    # 支援 market、limit、stop market
    type_combo.addItems(["market", "limit", "stop market"]) 

    lbl_price = QtWidgets.QLabel("Price")
    price_edit = QtWidgets.QLineEdit()
    price_edit.setPlaceholderText("e.g. 3500.0 (optional for market)")

    lbl_qty = QtWidgets.QLabel("Quantity")
    qty_spin = QtWidgets.QDoubleSpinBox()
    qty_spin.setDecimals(4)
    qty_spin.setMinimum(0.0)
    qty_spin.setMaximum(1_000_000)
    qty_spin.setSingleStep(1.0)
    qty_spin.setValue(1.0)

    btn_pos_1_3 = QtWidgets.QPushButton("1/3 P")
    btn_pos_1_2 = QtWidgets.QPushButton("1/2 P")
    btn_pos_full = QtWidgets.QPushButton("full P")
    for _btn in (btn_pos_1_3, btn_pos_1_2, btn_pos_full):
        _btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    pos_fill_row = QtWidgets.QWidget()
    pos_fill_layout = QtWidgets.QHBoxLayout(pos_fill_row)
    pos_fill_layout.setContentsMargins(0, 0, 0, 0)
    pos_fill_layout.setSpacing(6)
    pos_fill_layout.addWidget(btn_pos_1_3, 1)
    pos_fill_layout.addWidget(btn_pos_1_2, 1)
    pos_fill_layout.addWidget(btn_pos_full, 1)

    btn_place = QtWidgets.QPushButton("Place Order")
    btn_close = QtWidgets.QPushButton("Close Position")

    # 掛單表格與取消
    orders_table = QtWidgets.QTableWidget()
    orders_table.setColumnCount(6)
    orders_table.setHorizontalHeaderLabels(["ID", "Side", "Type", "Price", "Qty", "Status"])
    orders_table.horizontalHeader().setStretchLastSection(True)
    orders_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    orders_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    btn_cancel_selected = QtWidgets.QPushButton("Cancel Selected")

    # 成交明細表格
    lbl_trades = QtWidgets.QLabel("Trade History")
    trades_table = QtWidgets.QTableWidget()
    trades_table.setColumnCount(7)
    trades_table.setHorizontalHeaderLabels(["Time", "Side", "Price", "Qty", "PnL", "R_PnL", "Action"])
    trades_table.horizontalHeader().setStretchLastSection(True)
    trades_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    trades_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    lbl_unreal = QtWidgets.QLabel("Unrealized PnL: 0.00")
    lbl_real = QtWidgets.QLabel("Realized PnL: 0.00")
    lbl_pos = QtWidgets.QLabel("Position: flat")

    # 佈局
    order_panel.addWidget(lbl_side, row=0, col=0)
    order_panel.addWidget(side_combo, row=0, col=1)
    order_panel.addWidget(lbl_type, row=1, col=0)
    order_panel.addWidget(type_combo, row=1, col=1)
    order_panel.addWidget(lbl_price, row=2, col=0)
    order_panel.addWidget(price_edit, row=2, col=1)
    order_panel.addWidget(lbl_qty, row=3, col=0)
    order_panel.addWidget(qty_spin, row=3, col=1)
    order_panel.addWidget(pos_fill_row, row=4, col=0, colspan=2)
    order_panel.addWidget(btn_place, row=5, col=0, colspan=2)
    order_panel.addWidget(btn_close, row=6, col=0, colspan=2)
    order_panel.addWidget(lbl_unreal, row=7, col=0, colspan=2)
    order_panel.addWidget(lbl_real, row=8, col=0, colspan=2)
    order_panel.addWidget(lbl_pos, row=9, col=0, colspan=2)
    order_panel.addWidget(orders_table, row=10, col=0, colspan=2)
    order_panel.addWidget(btn_cancel_selected, row=11, col=0, colspan=2)
    order_panel.addWidget(lbl_trades, row=12, col=0, colspan=2)
    order_panel.addWidget(trades_table, row=13, col=0, colspan=2)
    
    btn_export_trades = QtWidgets.QPushButton("Export Trade History")
    btn_clean_trades = QtWidgets.QPushButton("Clean Trade History")
    btn_export_trades.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    btn_clean_trades.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    history_btn_row = QtWidgets.QWidget()
    history_btn_layout = QtWidgets.QHBoxLayout(history_btn_row)
    history_btn_layout.setContentsMargins(0, 0, 0, 0)
    history_btn_layout.setSpacing(6)
    history_btn_layout.addWidget(btn_export_trades, 1)
    history_btn_layout.addWidget(btn_clean_trades, 1)
    order_panel.addWidget(history_btn_row, row=14, col=0, colspan=2)

    # 狀態：部位 / 未成交委託 / 已實現 PnL / 成交紀錄
    position = None  # {"side": "long"|"short", "entry": float, "qty": float}
    realized_pnl = 0.0
    realized_r_pnl = 0.0
    pending_orders = []  # list of {id, side, type, price, qty, status}
    order_id_seq = 1
    trade_history = []  # list of {time, side, price, qty, pnl, r_pnl, action}
    trade_markers = []  # 成交標記（三角形）圖形物件
    trade_marker_events = []  # list of {timestamp, price, side, action, source_timeframe, origin}

    def current_price():
        # 目前已繪製最後一根的收盤價作為成交基準
        if idx > replay_start_idx:
            i = max(0, idx - 1)
            return float(df.iloc[i]["close"])
        return None

    def prefill_order_by_axis_buttons(target_side: str):
        """Fill the order panel values only (no submit) using last close vs axis box price."""
        cp = current_price()
        if cp is None:
            QtWidgets.QMessageBox.warning(None, "Order", "尚無K線，無法設定訂單")
            return

        # 方形價格以主圖右軸價格方形（hline）為準
        box_price = float(hline.value())

        if cp > box_price:
            # 收盤價高於方形價格
            order_type = "stop market" if target_side == "short" else "limit"
        else:
            # 收盤價低於（或等於）方形價格
            order_type = "limit" if target_side == "short" else "stop market"

        side_combo.setCurrentText(target_side)
        type_combo.setCurrentText(order_type)
        price_prec = int(df.iloc[max(0, idx - 1)].get("price_precision", price_precision_default))
        price_edit.setText(f"{box_price:.{price_prec}f}")

    def fill_qty_by_position_ratio(ratio: float):
        base_qty = float(position["qty"]) if position is not None else 0.0
        qty_spin.setValue(base_qty * ratio)

    def side_mult(side: str) -> int:
        return 1 if side == "long" else -1

    def update_pnl_labels():
        cp = current_price()
        unreal = 0.0
        unreal_r = 0.0
        if position is not None and cp is not None:
            unreal = (cp - position["entry"]) * side_mult(position["side"]) * position["qty"]
            unreal_r = unreal / max_SL
        lbl_unreal.setText(f"Unrealized PnL: {unreal:.3f} ({unreal_r:.3f}R)")
        lbl_real.setText(f"Realized PnL: {realized_pnl:.3f} ({realized_r_pnl:.3f}R)")
        if position is None:
            lbl_pos.setText("Position: flat")
        else:
            lbl_pos.setText(
                f"Position: {position['side']} {position['qty']:.4f} @ {position['entry']:.{price_precision_default}f}"
            )

    def refresh_orders_table():
        orders_table.setRowCount(len(pending_orders))
        for r, od in enumerate(pending_orders):
            orders_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(od['id'])))
            orders_table.setItem(r, 1, QtWidgets.QTableWidgetItem(od['side']))
            orders_table.setItem(r, 2, QtWidgets.QTableWidgetItem(od['type']))
            orders_table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{od['price']:.{price_precision_default}f}"))
            orders_table.setItem(r, 4, QtWidgets.QTableWidgetItem(f"{od['qty']:.4f}"))
            orders_table.setItem(r, 5, QtWidgets.QTableWidgetItem(od['status']))

    def refresh_trades_table():
        trades_table.setRowCount(len(trade_history))
        for r, trade in enumerate(trade_history):
            trades_table.setItem(r, 0, QtWidgets.QTableWidgetItem(trade['time']))
            trades_table.setItem(r, 1, QtWidgets.QTableWidgetItem(trade['side']))
            trades_table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{trade['price']:.{price_precision_default}f}"))
            trades_table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{trade['qty']:.4f}"))
            pnl_text = f"{trade['pnl']:.3f}" if trade['pnl'] != 0 else "-"
            trades_table.setItem(r, 4, QtWidgets.QTableWidgetItem(pnl_text))
            r_pnl_text = f"{trade['r_pnl']:.3f}R" if trade['r_pnl'] != 0 else "-"
            trades_table.setItem(r, 5, QtWidgets.QTableWidgetItem(r_pnl_text))
            trades_table.setItem(r, 6, QtWidgets.QTableWidgetItem(trade['action']))

    def fill_order(
        fill_side: str,
        fill_price: float,
        fill_qty: float,
        fill_timestamp: float = 0,
        fill_time_str: str = "",
        marker_origin: str = "market",
    ):
        nonlocal position, realized_pnl, realized_r_pnl
        fprice = float(fill_price)
        fqty = float(fill_qty)
        if fqty <= 0:
            return
        
        current_time = fill_time_str
        current_ts = fill_timestamp
        
        if position is None:
            position = {"side": fill_side, "entry": fprice, "qty": fqty}
            # 記錄開倉
            trade_history.append({
                "time": current_time,
                "side": fill_side,
                "price": fprice,
                "qty": fqty,
                "pnl": 0.0,
                "r_pnl": 0.0,
                "action": "OPEN"
            })
            # 在圖表上添加三角形標記
            add_trade_marker(current_ts, fprice, fill_side, "OPEN", marker_origin)
        else:
            if position["side"] == fill_side:
                # 同向加倉：以加權平均價格更新
                new_qty = position["qty"] + fqty
                position["entry"] = (position["entry"] * position["qty"] + fprice * fqty) / new_qty
                position["qty"] = new_qty
                # 記錄加倉
                trade_history.append({
                    "time": current_time,
                    "side": fill_side,
                    "price": fprice,
                    "qty": fqty,
                    "pnl": 0.0,
                    "r_pnl": 0.0,
                    "action": "ADD"
                })
                # 在圖表上添加三角形標記
                add_trade_marker(current_ts, fprice, fill_side, "ADD", marker_origin)
            else:
                # 反向下單：先抵消
                close_qty = min(position["qty"], fqty)
                pnl = (fprice - position["entry"]) * side_mult(position["side"]) * close_qty
                r_pnl = pnl / max_SL
                realized_pnl += pnl
                realized_r_pnl += r_pnl
                # 記錄平倉
                trade_history.append({
                    "time": current_time,
                    "side": fill_side,
                    "price": fprice,
                    "qty": close_qty,
                    "pnl": pnl,
                    "r_pnl": r_pnl,
                    "action": "CLOSE"
                })
                # 在圖表上添加三角形標記
                add_trade_marker(current_ts, fprice, fill_side, "CLOSE", marker_origin)
                remaining = fqty - close_qty
                if position["qty"] > close_qty:
                    # 原部位剩餘
                    position["qty"] -= close_qty
                    if position["qty"] <= 0:
                        position = None
                else:
                    # 原部位全平
                    position = None
                if remaining > 0:
                    # 剩餘的以新方向開倉
                    position = {"side": fill_side, "entry": fprice, "qty": remaining}
                    trade_history.append({
                        "time": current_time,
                        "side": fill_side,
                        "price": fprice,
                        "qty": remaining,
                        "pnl": 0.0,
                        "r_pnl": 0.0,
                        "action": "OPEN"
                    })
                    # 在圖表上添加三角形標記（反手開倉）
                    add_trade_marker(current_ts, fprice, fill_side, "OPEN", marker_origin)
        refresh_trades_table()
        update_pnl_labels()

    def place_order():
        nonlocal order_id_seq
        s = side_combo.currentText()
        t = type_combo.currentText()
        p_text = price_edit.text().strip()
        q_val = float(qty_spin.value())

        cp = current_price()
        if cp is None:
            QtWidgets.QMessageBox.warning(None, "Order", "尚無K線，無法下單")
            return

        if t == "market":
            # 市價單立即以目前價成交
            fill_ts = 0
            fill_time = ""
            if idx > replay_start_idx:
                fill_ts = df.iloc[idx-1]["dt_based_ts"]
                fill_time = str(df.iloc[idx-1]["dt_based"]).split('.')[0]
            fill_order(s, cp, q_val, fill_ts, fill_time, marker_origin="market")
            return

        # 非市價需要價格
        try:
            price_val = float(p_text)
        except Exception:
            QtWidgets.QMessageBox.warning(None, "Order", "請輸入有效價格")
            return

        order = {
            "id": order_id_seq,
            "side": s,
            "type": t,  # "limit" or "stop market"
            "price": price_val,
            "qty": q_val,
            "status": "open",
        }
        order_id_seq += 1
        pending_orders.append(order)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"已送出 {t} {s} @ {price_val}")
        refresh_orders_table()

    def close_position():
        nonlocal position, realized_pnl, realized_r_pnl
        if position is None:
            return
        cp = current_price()
        if cp is None:
            return
        pnl = (cp - position["entry"]) * side_mult(position["side"]) * position["qty"]
        r_pnl = pnl / max_SL
        realized_pnl += pnl
        realized_r_pnl += r_pnl
        
        # 記錄平倉
        fill_ts = 0
        fill_time = ""
        if idx > replay_start_idx:
            fill_ts = df.iloc[idx-1]["dt_based_ts"]
            fill_time = str(df.iloc[idx-1]["dt_based"]).split('.')[0]
        close_side = "short" if position["side"] == "long" else "long"
        trade_history.append({
            "time": fill_time,
            "side": close_side,
            "price": cp,
            "qty": position["qty"],
            "pnl": pnl,
            "r_pnl": r_pnl,
            "action": "CLOSE"
        })
        # 在圖表上添加三角形標記
        add_trade_marker(fill_ts, cp, close_side, "CLOSE", marker_origin="manual-close")
        
        position = None
        refresh_trades_table()
        update_pnl_labels()

    def cancel_selected_orders():
        selected = orders_table.selectionModel().selectedRows()
        if not selected:
            return
        ids = sorted([int(orders_table.item(r.row(), 0).text()) for r in selected], reverse=True)
        for oid in ids:
            for i, od in enumerate(list(pending_orders)):
                if od.get('id') == oid and od.get('status') == 'open':
                    pending_orders.pop(i)
                    break
        refresh_orders_table()

    def export_trade_history():
        if not trade_history:
            QtWidgets.QMessageBox.information(None, "Export", "No trade history to export")
            return
        
        # 使用檔案對話框選擇儲存位置
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, 
            "Export Trade History", 
            "trade_history.csv", 
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if not file_path:
            return
        
        try:
            # 轉換為 DataFrame
            df_trades = pd.DataFrame(trade_history)
            
            if file_path.endswith('.xlsx'):
                df_trades.to_excel(file_path, index=False)
            else:
                df_trades.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            QtWidgets.QMessageBox.information(None, "Export", f"Trade history exported to:\n{file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Export Error", f"Failed to export:\n{str(e)}")

    def clean_trade_history():
        nonlocal position, realized_pnl, realized_r_pnl, order_id_seq
        if trade_markers:
            for marker in trade_markers:
                try:
                    plot.removeItem(marker)
                except Exception:
                    pass

        position = None
        realized_pnl = 0.0
        realized_r_pnl = 0.0
        pending_orders.clear()
        order_id_seq = 1

        trade_markers.clear()
        trade_marker_events.clear()
        trade_history.clear()
        refresh_orders_table()
        refresh_trades_table()
        update_pnl_labels()
    
    btn_place.clicked.connect(place_order)
    btn_close.clicked.connect(close_position)
    btn_cancel_selected.clicked.connect(cancel_selected_orders)
    btn_export_trades.clicked.connect(export_trade_history)
    btn_clean_trades.clicked.connect(clean_trade_history)
    btn_short.clicked.connect(lambda: prefill_order_by_axis_buttons("short"))
    btn_long.clicked.connect(lambda: prefill_order_by_axis_buttons("long"))
    btn_pos_1_3.clicked.connect(lambda: fill_qty_by_position_ratio(1.0 / 3.0))
    btn_pos_1_2.clicked.connect(lambda: fill_qty_by_position_ratio(1.0 / 2.0))
    btn_pos_full.clicked.connect(lambda: fill_qty_by_position_ratio(1.0))

    def on_market_selection_changed(_text: str):
        nonlocal coin_name, timeframe
        nonlocal df, idx, replay_start_idx, price_precision_default
        nonlocal position, realized_pnl, realized_r_pnl, order_id_seq
        nonlocal line_mode, line_start, temp_line, selected_item, fibo_mode, fibo_base_price, text_mode, range_mode, range_start_price
        nonlocal auto_all_mode
        nonlocal visible_first_dt_ny, visible_last_dt_ny

        next_coin = product_combo.currentText().strip()
        next_timeframe = timeframe_combo.currentText().strip()
        if next_timeframe not in timeframe_options:
            return

        if next_coin == coin_name and next_timeframe == timeframe:
            return

        refresh_visible_candle_record()
        target_first_dt_ny = visible_first_dt_ny
        target_last_dt_ny = visible_last_dt_ny

        timeframe_loading_label.setVisible(True)
        QtWidgets.QApplication.processEvents()

        try:
            next_df = load_market_dataframe(next_coin, next_timeframe)

            if target_first_dt_ny is not None and target_last_dt_ny is not None:
                next_first_idx = next_df[next_df["dt_ny"] >= target_first_dt_ny].index.min()
                if pd.isna(next_first_idx):
                    next_first_idx = 0

                next_last_idx = next_df[next_df["dt_ny"] >= target_last_dt_ny].index.min()
                if pd.isna(next_last_idx):
                    next_last_idx = len(next_df) - 1

                next_replay_start_idx = int(next_first_idx)
                next_idx = int(max(next_replay_start_idx + 1, int(next_last_idx) + 1))
                next_idx = min(next_idx, len(next_df))
            else:
                next_replay_start_idx = get_replay_start_index(next_df)
                next_idx = next_replay_start_idx
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "切換失敗", f"無法載入 {next_coin} {next_timeframe} 資料\n{e}")
            product_combo.blockSignals(True)
            timeframe_combo.blockSignals(True)
            product_combo.setCurrentText(coin_name)
            timeframe_combo.setCurrentText(timeframe)
            product_combo.blockSignals(False)
            timeframe_combo.blockSignals(False)
            timeframe_loading_label.setVisible(False)
            return

        is_product_changed = next_coin != coin_name

        coin_name = next_coin
        timeframe = next_timeframe

        df = next_df
        set_time_base(time_base_name)
        replay_start_idx = int(next_replay_start_idx)
        idx = int(next_idx)
        price_precision_default = int(df["price_precision"].iloc[0]) if len(df) > 0 else 2

        # 切換商品/週期後，重置 replay 相關狀態
        auto_all_mode = False
        if is_product_changed:
            # 商品切換時才清部位與掛單；單純 timeframe 切換需保留
            position = None
            realized_pnl = 0.0
            realized_r_pnl = 0.0
            pending_orders.clear()
            trade_markers.clear()
            trade_marker_events.clear()
            order_id_seq = 1
            saved_lines.clear()
            fibo_groups.clear()
            line_custom_pens.clear()
            price_ranges.clear()
            line_tool_manager.clear_all()
        line_mode = False
        line_start = None
        temp_line = None
        selected_item = None
        fibo_mode = False
        fibo_base_price = None
        fibo_manager.cancel_mode()
        text_mode = False
        range_mode = False
        range_start_price = None

        title_text = f"{coin_name} Replay UI {timeframe}"
        root.setWindowTitle(title_text)
        win.setWindowTitle(title_text)

        refresh_orders_table()
        refresh_trades_table()
        redraw_all()

        if idx > replay_start_idx:
            one_tick_sec = float(timeframe_to_seconds(timeframe))
            x_left = float(df.iloc[replay_start_idx]["dt_based_ts"]) - one_tick_sec
            x_right = float(df.iloc[idx - 1]["dt_based_ts"]) + one_tick_sec
            if x_right <= x_left:
                x_right = x_left + 1.0
            plot.setXRange(x_left, x_right, padding=0)

        update_pnl_labels()

        anchor_idx = min(max(idx - 1, replay_start_idx), len(df) - 1)
        update_indicator_value_panel(anchor_idx)
        update_kline_info_panel(anchor_idx)
        update_axis_value_boxes(anchor_idx)
        refresh_visible_candle_record()
        sync_jump_datetime_controls(anchor_idx)
        timeframe_loading_label.setVisible(False)

    product_combo.currentTextChanged.connect(on_market_selection_changed)
    timeframe_combo.currentTextChanged.connect(on_market_selection_changed)

    preload_queue = [(product, tf) for product in product_options for tf in ["1m", "5m"]]

    def preload_hot_market_data_step():
        if not preload_queue:
            return

        product, tf = preload_queue.pop(0)
        if (product, tf) not in market_df_cache:
            try:
                load_market_dataframe(product, tf)
                print(f"Preloaded dataframe: {product}_{tf}")
            except Exception as e:
                print(f"Preload skipped for {product}_{tf}. reason={e}")

        # 分批預載，避免啟動時阻塞 UI 造成白屏
        QtCore.QTimer.singleShot(50, preload_hot_market_data_step)

    QtCore.QTimer.singleShot(300, preload_hot_market_data_step)

    # ================================================================
    # 成交標記功能
    # ================================================================
    def draw_trade_marker_scatter(timestamp: float, price: float, side: str):
        if timestamp == 0 or price == 0:
            return None

        if side == "long":
            symbol = 't1'
            color = (0, 255, 0)
        else:
            symbol = 't'
            color = (255, 0, 0)

        scatter = pg.ScatterPlotItem(
            pos=[[timestamp, price]],
            size=8,
            symbol=symbol,
            brush=pg.mkBrush(color),
            pen=pg.mkPen('w', width=1)
        )
        plot.addItem(scatter)
        return scatter

    def remap_marker_timestamp_for_timeframe(event: dict) -> float:
        if len(df) == 0:
            return float(event["timestamp"])

        source_ts = float(event["timestamp"])
        if source_ts == 0:
            return 0.0

        source_tf = event.get("source_timeframe", timeframe)
        origin = event.get("origin", "market")
        price = float(event.get("price", 0.0))

        target_sec = float(timeframe_to_seconds(timeframe))
        source_sec = float(timeframe_to_seconds(source_tf)) if source_tf in timeframe_options else target_sec

        # 使用完整資料映射，避免切換多次時因當前 idx 視窗裁切導致標記消失
        sub_df = df
        ts_arr = sub_df["dt_based_ts"].to_numpy(dtype=float)
        if ts_arr.size == 0:
            return source_ts

        # 小 -> 大：往下對齊到目標週期整點（例如 12:05 -> 12:00）
        if target_sec >= source_sec:
            snapped = np.floor(source_ts / target_sec) * target_sec
            pos = int(np.searchsorted(ts_arr, snapped, side="left"))
            if pos >= ts_arr.size:
                pos = ts_arr.size - 1
            if pos > 0 and ts_arr[pos] > source_ts:
                pos -= 1
            return float(ts_arr[pos])

        # 大 -> 小：掛單成交標記優先找原大K區間內「第一根可成交」小K
        if origin == "pending":
            window_start = source_ts - source_sec
            candidates = sub_df[(sub_df["dt_based_ts"] > window_start) & (sub_df["dt_based_ts"] <= source_ts)]
            if len(candidates) > 0:
                touched = candidates[(candidates["low"] <= price) & (candidates["high"] >= price)]
                if len(touched) > 0:
                    return float(touched.iloc[0]["dt_based_ts"])

        pos = int(np.searchsorted(ts_arr, source_ts, side="left"))
        if pos >= ts_arr.size:
            pos = ts_arr.size - 1
        return float(ts_arr[pos])

    def rebuild_trade_markers_for_current_timeframe():
        trade_markers.clear()
        for event in trade_marker_events:
            mapped_ts = remap_marker_timestamp_for_timeframe(event)
            marker = draw_trade_marker_scatter(mapped_ts, float(event["price"]), str(event["side"]))
            if marker is not None:
                trade_markers.append(marker)

    def add_trade_marker(timestamp, price, side, action, marker_origin: str = "market"):
        """在圖表上添加成交三角形標記"""
        if timestamp == 0 or price == 0:
            return

        event = {
            "timestamp": float(timestamp),
            "price": float(price),
            "side": side,
            "action": action,
            "source_timeframe": timeframe,
            "origin": marker_origin,
        }
        trade_marker_events.append(event)

        mapped_ts = remap_marker_timestamp_for_timeframe(event)
        scatter = draw_trade_marker_scatter(mapped_ts, float(price), side)
        if scatter is not None:
            trade_markers.append(scatter)

    # MA 曲線
    ma_curve_20 = plot.plot(pen=pg.mkPen("green", width=2))
    ma_curve_100 = plot.plot(pen=pg.mkPen("orange", width=2))

    # 下方指標曲線（對齊 TradingView: cross + stepline）
    bull_bar_scatter = pg.ScatterPlotItem(
        size=8,
        symbol='x',
        pen=pg.mkPen((255, 204, 128), width=2),
        brush=None,
        name="bull_bar_score"
    )
    bear_bar_scatter = pg.ScatterPlotItem(
        size=8,
        symbol='x',
        pen=pg.mkPen((244, 143, 177), width=2),
        brush=None,
        name="bear_bar_score"
    )
    indicator_plot.addItem(bull_bar_scatter)
    indicator_plot.addItem(bear_bar_scatter)
    bull_run_curve = indicator_plot.plot(pen=pg.mkPen((8, 153, 129), width=2), name="bull_run")
    bear_run_curve = indicator_plot.plot(pen=pg.mkPen((229, 57, 53), width=2), name="bear_run")
    indicator_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen((255, 255, 255, 80), width=1)))

    # ================================================================
    # 畫 K 線 function
    # ================================================================
    def draw_bars_batch(sub_df: pd.DataFrame):
        if len(sub_df) == 0:
            return

        t = sub_df["dt_based_ts"].to_numpy(dtype=float)
        o = sub_df["open"].to_numpy(dtype=float)
        h = sub_df["high"].to_numpy(dtype=float)
        l = sub_df["low"].to_numpy(dtype=float)
        c = sub_df["close"].to_numpy(dtype=float)

        # Draw all wicks as independent (x0,y0)-(x1,y1) pairs to avoid cross-candle connecting lines.
        x_wick = np.repeat(t, 2)
        y_wick = np.empty(t.size * 2, dtype=float)
        y_wick[0::2] = l
        y_wick[1::2] = h
        plot.addItem(pg.PlotDataItem(x=x_wick, y=y_wick, connect='pairs', pen=pg.mkPen("white", width=1.3)))

        center_y = (o + c) * 0.5
        body_h = np.abs(c - o)
        body_w = 0.8 * float(timeframe_to_seconds(timeframe))
        bull_mask = c >= o
        bear_mask = ~bull_mask

        if bull_mask.any():
            plot.addItem(pg.BarGraphItem(
                x=t[bull_mask],
                width=body_w,
                height=body_h[bull_mask],
                y=center_y[bull_mask],
                brush=(31, 121, 245),
                pen=None,
            ))
        if bear_mask.any():
            plot.addItem(pg.BarGraphItem(
                x=t[bear_mask],
                width=body_w,
                height=body_h[bear_mask],
                y=center_y[bear_mask],
                brush=(247, 82, 95),
                pen=None,
            ))

    def draw_bar(record):
        t = record["dt_based"].timestamp()
        o, h, l, c = record["open"], record["high"], record["low"], record["close"]

        wick_pen = pg.mkPen("white", width=1.3)


        plot.addItem(pg.PlotCurveItem(
            x=[t, t],
            y=[l, h],
            pen=wick_pen
        ))

        body_color = (31, 121, 245) if c >= o else (247, 82, 95)

        body_top = max(o, c)
        body_bottom = min(o, c)
        body_height = body_top - body_bottom
        center_y = (body_top + body_bottom) / 2
        w = 0.4 * float(timeframe_to_seconds(timeframe))

        body = pg.BarGraphItem(
            x=[t],
            width=2 * w,
            height=[body_height],
            y=[center_y],
            brush=body_color,
            pen=None
        )
        plot.addItem(body)

    # 檢查委託是否觸發成交（以本根K的最高/最低判斷觸發）
    def process_pending_orders(current_bar):
        if not pending_orders:
            return
        high_p = float(current_bar["high"]) if "high" in current_bar else None
        low_p = float(current_bar["low"]) if "low" in current_bar else None
        # 取得當前K線的時間戳和時間字串
        fill_ts = float(current_bar["dt_based_ts"]) if "dt_based_ts" in current_bar else 0
        fill_time = str(current_bar["dt_based"]).split('.')[0] if "dt_based" in current_bar else ""
        to_close = []
        for od in pending_orders:
            if od.get("status") != "open":
                continue
            s = od["side"]
            typ = od["type"]
            px = float(od["price"])
            qv = float(od["qty"])
            filled = False
            # 只有當委託價格介於該根K線的高低點之間（有碰到）才成交
            if low_p is not None and high_p is not None:
                touched = (low_p <= px <= high_p)
                if typ in ("limit", "stop market") and touched:
                    filled = True
            if filled:
                od["status"] = "filled"
                fill_order(s, px, qv, fill_ts, fill_time, marker_origin="pending")
                to_close.append(od)
        # 清除已成交
        if to_close:
            for od in to_close:
                try:
                    pending_orders.remove(od)
                except ValueError:
                    pass
            refresh_orders_table()
    def draw_vertical_grids():
    # 先刪掉舊 grid
        for line in grid_lines:
            plot.removeItem(line)
        grid_lines.clear()

        # 根據 idx 畫新的 grid
        for i in range(replay_start_idx, idx):
            minute = df.iloc[i]["dt_based"].minute
            if minute not in (0, 15, 30, 45):
                continue

            ts = df.iloc[i]["dt_based_ts"]
            line = pg.InfiniteLine(
                ts, angle=90,
                pen=pg.mkPen((255, 255, 255, 40), width=1)
            )
            plot.addItem(line)
            grid_lines.append(line)

    # ================================================================
    # redraw（倒帶）
    # ================================================================
    def redraw_all():
        def set_scatter_by_mask(scatter_obj, x_vals, y_vals, mask_vals):
            x_arr = np.asarray(x_vals, dtype=float)
            y_arr = np.asarray(y_vals, dtype=float)
            mask_arr = np.asarray(mask_vals, dtype=bool)
            if x_arr.size == 0 or y_arr.size == 0 or not mask_arr.any():
                scatter_obj.setData(x=[], y=[])
                return
            scatter_obj.setData(x=x_arr[mask_arr], y=y_arr[mask_arr])

        plot.clear()
        indicator_plot.clear()

        nonlocal ma_curve_20, ma_curve_100, bull_run_curve, bear_run_curve, bull_bar_scatter, bear_bar_scatter
        ma_curve_20 = plot.plot(pen=pg.mkPen("green", width=2))
        if timeframe == "1m":
            ma_curve_100 = plot.plot(pen=pg.mkPen("orange", width=2))
        bull_bar_scatter = pg.ScatterPlotItem(
            size=8,
            symbol='x',
            pen=pg.mkPen((255, 204, 128), width=2),
            brush=None,
            name="bull_bar_score"
        )
        bear_bar_scatter = pg.ScatterPlotItem(
            size=8,
            symbol='x',
            pen=pg.mkPen((244, 143, 177), width=2),
            brush=None,
            name="bear_bar_score"
        )
        indicator_plot.addItem(bull_bar_scatter)
        indicator_plot.addItem(bear_bar_scatter)
        bull_run_curve = indicator_plot.plot(pen=pg.mkPen((8, 153, 129), width=2), name="bull_run")
        bear_run_curve = indicator_plot.plot(pen=pg.mkPen((229, 57, 53), width=2), name="bear_run")
        indicator_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen((255, 255, 255, 80), width=1)))

        if idx > replay_start_idx:
            shown = df.iloc[replay_start_idx:idx]
            draw_bars_batch(shown)

            # Cap text items to keep redraw responsive on large 1m windows.
            label_step = max(1, len(shown) // 240)
            for i in range(replay_start_idx, idx, label_step):
                bar = df.iloc[i]
                offset = (bar["high"] - bar["low"]) * 0.3
                txt = pg.TextItem(
                    text=str(i - replay_start_idx + 1),
                    color="white",
                    anchor=(0.5, 1.0)
                )
                txt.setPos(float(bar["dt_based_ts"]), bar["low"] - offset)
                plot.addItem(txt)

        if idx > replay_start_idx:
            ma_vals = df.iloc[replay_start_idx:idx]["MA20"].values
            ma_vals_100 = df.iloc[replay_start_idx:idx]["MA100"].values
            times = df.iloc[replay_start_idx:idx]["dt_based_ts"].values
            ma_curve_20.setData(times, ma_vals)
            if timeframe == "1m":
                ma_curve_100.setData(times, ma_vals_100)

            bull_mask = (df.iloc[replay_start_idx:idx]["close"] > df.iloc[replay_start_idx:idx]["open"]).values
            bear_mask = (df.iloc[replay_start_idx:idx]["close"] < df.iloc[replay_start_idx:idx]["open"]).values
            set_scatter_by_mask(
                bull_bar_scatter,
                times,
                df.iloc[replay_start_idx:idx]["bull_bar_score"].values,
                bull_mask,
            )
            set_scatter_by_mask(
                bear_bar_scatter,
                times,
                df.iloc[replay_start_idx:idx]["bear_bar_score"].values,
                bear_mask,
            )
            bull_step_x, bull_step_y = centered_step_xy(times, df.iloc[replay_start_idx:idx]["bull_run"].values)
            bear_step_x, bear_step_y = centered_step_xy(times, df.iloc[replay_start_idx:idx]["bear_run"].values)
            bull_run_curve.setData(bull_step_x, bull_step_y)
            bear_run_curve.setData(bear_step_x, bear_step_y)

        plot.addItem(vline, ignoreBounds=True)
        plot.addItem(hline, ignoreBounds=True)
        indicator_plot.addItem(vline_indicator, ignoreBounds=True)
        indicator_plot.addItem(hline_indicator, ignoreBounds=True)
        set_crosshair_visibility(current_crosshair_active)
        draw_vertical_grids()
        
        # 重繪成交標記（依當前 timeframe 重新映射 X 位置）
        rebuild_trade_markers_for_current_timeframe()
        
        # 重繪保存的劃線
        for line in saved_lines:
            plot.addItem(line)
        
        # 重繪後更新 PnL 顯示
        update_pnl_labels()

        if idx > replay_start_idx:
            update_indicator_axis_range(max(replay_start_idx, idx - 1))
        else:
            indicator_plot.setYRange(-15, 15, padding=0)

        if idx > replay_start_idx:
            update_indicator_value_panel(max(replay_start_idx, idx - 1))

        if len(df) > 0:
            anchor_idx = replay_start_idx if idx <= replay_start_idx else min(idx - 1, len(df) - 1)
            update_kline_info_panel(max(0, anchor_idx))

        if auto_all_mode:
            apply_auto_all_view()

        refresh_visible_candle_record()

    def update_kline_info_panel(row_idx: int):
        if len(df) == 0:
            return

        row_idx = min(max(row_idx, 0), len(df) - 1)
        row = df.iloc[row_idx]
        price_prec = int(row.get("price_precision", price_precision_default))

        def pfmt(v: float) -> str:
            return f"{float(v):.{price_prec}f}"

        if row_idx > 0:
            prev_close = float(df.iloc[row_idx - 1]["close"])
            delta_abs = float(row["close"]) - prev_close
            delta_pct = (delta_abs / prev_close * 100.0) if prev_close != 0 else 0.0
        else:
            delta_abs = 0.0
            delta_pct = 0.0

        delta_abs_text = f"{delta_abs:+.{price_prec}f}"
        delta_pct_text = f"{delta_pct:+.2f}"

        if delta_abs > 0:
            ohlc_color = "#1f79f5"
        elif delta_abs < 0:
            ohlc_color = "#f7525f"
        else:
            ohlc_color = "#cfd7e6"

        kline_ohlc_label.setStyleSheet(f"color: {ohlc_color}; font-size: 12pt; font-weight: 700;")
        kline_ohlc_label.setText(
            f"開={pfmt(row['open'])}   "
            f"高={pfmt(row['high'])}   "
            f"低={pfmt(row['low'])}   "
            f"收={pfmt(row['close'])}   "
            f"{delta_abs_text}({delta_pct_text}%)"
        )

        ema20_val = float(row["MA20"]) if pd.notna(row["MA20"]) else float("nan")
        if np.isfinite(ema20_val):
            kline_ema_label.setText(f"EMA 20={pfmt(ema20_val)}")
        else:
            kline_ema_label.setText("EMA 20=--")

    def get_visible_subset(max_row_idx: int) -> pd.DataFrame:
        upper = min(max_row_idx, len(df) - 1)
        if upper < replay_start_idx:
            return df.iloc[0:0]

        base = df.iloc[replay_start_idx:upper + 1]
        x0, x1 = plot.viewRange()[0]
        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        vis = base[(base["dt_based_ts"] >= x_min) & (base["dt_based_ts"] <= x_max)]
        if len(vis) > 0:
            return vis

        # 視窗內沒有點時，回退到距離視窗中心最近的一根
        center_x = (x_min + x_max) * 0.5
        nearest_idx = (base["dt_based_ts"] - center_x).abs().idxmin()
        return base.loc[[nearest_idx]]

    def update_main_axis_range(row_idx: int):
        if row_idx < 0:
            return

        sub = get_visible_subset(row_idx)
        if len(sub) == 0:
            return

        low_min = float(sub["low"].min())
        high_max = float(sub["high"].max())
        span = high_max - low_min
        pad = max(span * 0.08, max(abs(high_max), abs(low_min), 1.0) * 0.002)
        plot.setYRange(low_min - pad, high_max + pad, padding=0)

    def update_indicator_axis_range(row_idx: int):
        if row_idx < 0:
            indicator_plot.setYRange(-15, 15, padding=0)
            return

        sub = get_visible_subset(row_idx)
        bull_vals = sub["bull_run"].to_numpy(dtype=float)
        bear_vals = sub["bear_run"].to_numpy(dtype=float)
        bull_finite = bull_vals[np.isfinite(bull_vals)]
        bear_finite = bear_vals[np.isfinite(bear_vals)]

        bull_max = float(np.max(bull_finite)) if bull_finite.size > 0 else 0.0
        bear_min = float(np.min(bear_finite)) if bear_finite.size > 0 else 0.0

        y_max = bull_max + 3.0
        y_min = bear_min - 3.0
        if abs(y_max - y_min) < 1e-9:
            y_max = y_min + 1.0
        indicator_plot.setYRange(y_min, y_max, padding=0.03)

    def update_indicator_value_panel(row_idx: int):
        if row_idx < 0 or row_idx >= len(df):
            return

        row = df.iloc[row_idx]
        bull_active = row["close"] > row["open"]
        bear_active = row["close"] < row["open"]

        def fixed_slot(val: float, active: bool) -> str:
            txt = " #.# " if not active else f"{float(val):.1f}"
            if len(txt) > 6:
                txt = txt[:6]
            return txt.rjust(6).replace(" ", "&nbsp;")

        bull_bar_txt = fixed_slot(row["bull_bar_score"], bull_active)
        bear_bar_txt = fixed_slot(row["bear_bar_score"], bear_active)
        bull_run_txt = fixed_slot(row["bull_run"], bull_active)
        bear_run_txt = fixed_slot(row["bear_run"], bear_active)

        indicator_info_label.setText(
            "<span style='font-size:12pt; color:#cfd7e6;'>SBS Run Score&nbsp;&nbsp;</span>"
            f"<span style='font-size:12pt; color:#ffcc80;'>bull bar={bull_bar_txt}</span>"
            "<span style='font-size:12pt; color:#7f8897;'>&nbsp;&nbsp;</span>"
            f"<span style='font-size:12pt; color:#089981;'>bull run={bull_run_txt}</span>"
            "<span style='font-size:12pt; color:#7f8897;'>&nbsp;&nbsp;</span>"
            f"<span style='font-size:12pt; color:#f48fb1;'>bear bar={bear_bar_txt}</span>"
            "<span style='font-size:12pt; color:#7f8897;'>&nbsp;&nbsp;</span>"
            f"<span style='font-size:12pt; color:#e53935;'>bear run={bear_run_txt}</span>"
        )

    # Auto 按鈕或其他 range 變更後，強制套回自訂規則
    syncing_indicator_range = False

    def enforce_indicator_range_rules(*_args):
        nonlocal syncing_indicator_range
        if syncing_indicator_range:
            return
        if len(df) == 0:
            return

        syncing_indicator_range = True
        try:
            anchor_idx = min(max(idx - 1, replay_start_idx), len(df) - 1)
            update_indicator_axis_range(anchor_idx)
            update_indicator_value_panel(anchor_idx)
        finally:
            syncing_indicator_range = False

    indicator_plot.sigRangeChanged.connect(enforce_indicator_range_rules)

    def on_main_view_range_changed(*_args):
        nonlocal auto_all_mode
        if syncing_auto_all_range:
            return
        if auto_all_mode:
            auto_all_mode = False
        anchor_idx = min(max(idx - 1, replay_start_idx), len(df) - 1) if len(df) > 0 else 0
        update_axis_value_boxes(anchor_idx)
        refresh_visible_candle_record()

    plot.sigRangeChanged.connect(on_main_view_range_changed)

    def on_indicator_view_range_changed(*_args):
        anchor_idx = min(max(idx - 1, replay_start_idx), len(df) - 1) if len(df) > 0 else 0
        update_axis_value_boxes(anchor_idx)

    indicator_plot.sigRangeChanged.connect(on_indicator_view_range_changed)

    # ================================================================
    # 滑鼠移動
    # ================================================================
    def get_nearest_appeared_index_by_x(x_val: float) -> int:
        appeared_end = min(max(idx, replay_start_idx), len(df))
        sub = df.iloc[replay_start_idx:appeared_end]
        if len(sub) == 0:
            return replay_start_idx
        return int((sub["dt_based_ts"] - x_val).abs().idxmin())

    def update_axis_value_boxes(row_idx: int):
        if len(df) == 0:
            return

        row_idx = min(max(row_idx, 0), len(df) - 1)
        price_prec = int(df.iloc[row_idx].get("price_precision", price_precision_default))
        price_y = float(hline.value())
        indicator_y = float(hline_indicator.value())
        snapped_x = float(df.iloc[row_idx]["dt_based_ts"])

        # Main Y label on right axis band (overlay the axis area)
        main_axis_rect = plot.getAxis("right").sceneBoundingRect()
        main_vr = plot.vb.viewRange()
        main_ref_x = (main_vr[0][0] + main_vr[0][1]) * 0.5
        main_scene_pt = plot.vb.mapViewToScene(QtCore.QPointF(main_ref_x, price_y))
        main_widget_pt = win.mapFromScene(main_scene_pt)
        _ = main_axis_rect
        main_axis_left = max(0, win.width() - right_axis_width)
        main_axis_right = win.width() - 1

        price_axis_value_label.setText(f"{price_y:.{price_prec}f}")
        price_axis_value_label.setMinimumWidth(0)
        price_axis_value_label.setMaximumWidth(16777215)
        price_axis_value_label.adjustSize()
        p_w = price_axis_value_label.width()
        p_h = price_axis_value_label.height()
        p_x = main_axis_left + 2
        if p_x + p_w > main_axis_right - 1:
            p_x = max(main_axis_left + 1, main_axis_right - p_w - 1)
        p_y = int(main_widget_pt.y() - p_h * 0.5)
        p_x = max(0, min(p_x, win.width() - p_w))
        p_y = max(0, min(p_y, win.height() - p_h))
        price_axis_value_label.move(p_x, p_y)

        # Update [S] and [L] buttons: place them side-by-side and vertically centered with price label
        btn_size = 24
        btn_spacing = 2
        total_btn_w = btn_size * 2 + btn_spacing
        btn_start_x = max(0, p_x - total_btn_w - 2)
        btn_y = int(p_y + (p_h - btn_size) * 0.5)
        btn_y = max(0, min(btn_y, win.height() - btn_size))

        btn_short.move(btn_start_x, btn_y)
        btn_long.move(btn_start_x + btn_size + btn_spacing, btn_y)

        # Indicator Y label on right axis band (overlay the axis area)
        ind_axis_rect = indicator_plot.getAxis("right").sceneBoundingRect()
        ind_vr = indicator_plot.vb.viewRange()
        ind_ref_x = (ind_vr[0][0] + ind_vr[0][1]) * 0.5
        ind_scene_pt = indicator_plot.vb.mapViewToScene(QtCore.QPointF(ind_ref_x, indicator_y))
        ind_widget_pt = indicator_win.mapFromScene(ind_scene_pt)
        _ = ind_axis_rect
        ind_axis_left = max(0, indicator_win.width() - right_axis_width)
        ind_axis_right = indicator_win.width() - 1

        indicator_axis_value_label.setText(f"{indicator_y:.1f}")
        indicator_axis_value_label.setMinimumWidth(0)
        indicator_axis_value_label.setMaximumWidth(16777215)
        indicator_axis_value_label.adjustSize()
        i_w = indicator_axis_value_label.width()
        i_h = indicator_axis_value_label.height()
        i_x = ind_axis_left + 2
        if i_x + i_w > ind_axis_right - 1:
            i_x = max(ind_axis_left + 1, ind_axis_right - i_w - 1)
        i_y = int(ind_widget_pt.y() - i_h * 0.5)
        i_x = max(0, min(i_x, indicator_win.width() - i_w))
        i_y = max(0, min(i_y, indicator_win.height() - i_h))
        indicator_axis_value_label.move(i_x, i_y)

        dt = df.iloc[row_idx]["dt_based"]
        indicator_x_value_badge.setText(dt.strftime("%H:%M (%Y.%m.%d)"))

    def handle_crosshair_by_x(x_val: float, active: str, y_val: float):
        nonlocal current_crosshair_active
        nearest_idx = get_nearest_appeared_index_by_x(x_val)
        snapped_x = float(df.iloc[nearest_idx]["dt_based_ts"])

        vline.setPos(snapped_x)
        vline_indicator.setPos(snapped_x)

        current_crosshair_active = active
        set_crosshair_visibility(active)
        if active == "main":
            hline.setPos(y_val)
        else:
            snapped_y = quantize_step(y_val, 0.1)
            hline_indicator.setPos(snapped_y)

        update_indicator_value_panel(nearest_idx)
        update_kline_info_panel(nearest_idx)
        update_axis_value_boxes(nearest_idx)

    def mouseMovedMain(evt):
        nonlocal current_crosshair_active
        pos = evt[0]
        if not plot.vb.sceneBoundingRect().contains(pos):
            if current_crosshair_active == "main":
                current_crosshair_active = "none"
                set_crosshair_visibility("none")
            return
        mouse_point = plot.vb.mapSceneToView(pos)
        handle_crosshair_by_x(float(mouse_point.x()), "main", float(mouse_point.y()))

    def mouseMovedIndicator(evt):
        nonlocal current_crosshair_active
        pos = evt[0]
        if not indicator_plot.vb.sceneBoundingRect().contains(pos):
            if current_crosshair_active == "indicator":
                current_crosshair_active = "none"
                set_crosshair_visibility("none")
            return
        mouse_point = indicator_plot.vb.mapSceneToView(pos)
        handle_crosshair_by_x(float(mouse_point.x()), "indicator", float(mouse_point.y()))

    proxy_main = pg.SignalProxy(win.scene().sigMouseMoved, rateLimit=60, slot=mouseMovedMain)
    proxy_indicator = pg.SignalProxy(indicator_win.scene().sigMouseMoved, rateLimit=60, slot=mouseMovedIndicator)

    # ================================================================
    # 自訂文字輸入對話框
    # ================================================================
    class TextInputDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("輸入文字")
            self.resize(400, 300)
            
            layout = QtWidgets.QVBoxLayout(self)
            
            # 文字輸入區
            label = QtWidgets.QLabel("請輸入文字 (Shift+Enter 換行):")
            layout.addWidget(label)
            
            self.text_edit = QtWidgets.QTextEdit()
            self.text_edit.setMinimumHeight(100)
            layout.addWidget(self.text_edit)
            
            # 顏色選擇
            color_layout = QtWidgets.QHBoxLayout()
            color_label = QtWidgets.QLabel("顏色:")
            color_layout.addWidget(color_label)
            
            self.color_buttons = []
            colors = [
                ("", (255, 255, 255)),
                ("", (255, 80, 80)),
                ("", (255, 160, 0)),
                ("", (255, 240, 100)),
                ("", (100, 200, 100)),
                ("", (100, 180, 255)),
                ("", (200, 120, 255))
            ]
            
            self.color_group = QtWidgets.QButtonGroup(self)
            for i, (name, rgb) in enumerate(colors):
                btn = QtWidgets.QPushButton(name)
                btn.setFixedSize(50, 30)
                btn.setCheckable(True)
                btn.color_rgb = rgb
                btn.color_name = name
                # 預設未選中樣式
                btn.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 2px solid gray;")
                btn.toggled.connect(lambda checked, b=btn: self.update_color_button_style(b, checked))
                self.color_group.addButton(btn, i)
                color_layout.addWidget(btn)
                self.color_buttons.append(btn)
            
            self.color_buttons[0].setChecked(True)  # 預設白色
            color_layout.addStretch()
            layout.addLayout(color_layout)
            
            # 字體大小選擇
            size_layout = QtWidgets.QHBoxLayout()
            size_label = QtWidgets.QLabel("字體大小:")
            size_layout.addWidget(size_label)
            
            self.size_combo = QtWidgets.QComboBox()
            self.size_combo.addItems(["8", "10", "12", "14", "16", "18", "20", "24", "28", "32"])
            self.size_combo.setCurrentText("12")
            size_layout.addWidget(self.size_combo)
            size_layout.addStretch()
            layout.addLayout(size_layout)
            
            # 按鈕
            button_layout = QtWidgets.QHBoxLayout()
            ok_btn = QtWidgets.QPushButton("確定")
            cancel_btn = QtWidgets.QPushButton("取消")
            ok_btn.clicked.connect(self.accept)
            cancel_btn.clicked.connect(self.reject)
            button_layout.addStretch()
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
        
        def update_color_button_style(self, button, checked):
            """更新顏色按鈕樣式，選中時顯示粗外框"""
            rgb = button.color_rgb
            if checked:
                # 選中：粗黑框 + 陰影
                button.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 3px solid black; font-weight: bold;")
            else:
                # 未選中：細灰框
                button.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 2px solid gray;")
        
        def get_values(self):
            text = self.text_edit.toPlainText()
            checked_btn = self.color_group.checkedButton()
            color = checked_btn.color_rgb if checked_btn else (255, 255, 255)
            size = int(self.size_combo.currentText())
            return text, color, size

    # ================================================================
    # TradingView 風格色票
    # ================================================================
    def get_tradingview_color():
        """顯示 TradingView 風格的顏色選擇器"""
        dialog = QtWidgets.QColorDialog()
        
        # TradingView 基本色票（對應 Basic colors 區域，6行8列 = 48色）
        tv_basic_colors = [
            # Row 1 (was Col 1)
            (242, 54, 69), (252, 203, 205), (250, 161, 164), (247, 124, 128), (247, 82, 95),(242, 54, 69),
            # Row 2 (was Col 2)
            (255, 152, 0), (255, 224, 178), (255, 204, 128), (255, 183, 77), (255, 167, 38),(255, 152, 0),
            # Row 3 (was Col 3)
            (255, 235, 59), (255, 249, 196), (255, 245, 157), (255, 241, 118), (255, 238, 88),(255, 235, 59),
            # Row 4 (was Col 4)
            (76, 175, 80), (200, 230, 201), (165, 214, 167), (129, 199, 132), (102, 187, 106),(76, 175, 80),
            # Row 5 (was Col 5)
            (8, 153, 129), (172, 229, 220), (112, 204, 189), (66, 189, 168), (34, 171, 148),(8, 153, 129),
            # Row 6 (was Col 6)
            (0, 188, 212), (178, 235, 242), (128, 222, 234), (77, 208, 225), (38, 198, 218),(0, 188, 212),
            # Row 7 (was Col 7)
            (41, 98, 255), (187, 217, 251), (144, 191, 249), (91, 156, 246), (49, 121, 245), (41, 98, 255),
            # Row 8 (was Col 8)
            (103, 58, 183), (209, 196, 233), (179, 157, 219), (149, 117, 205), (126, 87, 194), (103, 58, 183)
        ]
        
        # 設定基本顏色（Basic colors 區域）
        for i, (r, g, b) in enumerate(tv_basic_colors):
            if i < 48:  # Qt 的 standard colors 有 48 個
                dialog.setStandardColor(i, QtGui.QColor(r, g, b))
        
        if dialog.exec():
            return dialog.currentColor()
        return None

    class LineSettingsDialog(QtWidgets.QDialog):
        STYLE_OPTIONS = [
            ("實線", "solid"),
            ("虛線", "dashed"),
            ("點線", "dotted"),
        ]

        def __init__(self, color_rgb, style_name, width_value, parent=None):
            super().__init__(parent)
            self.setWindowTitle("線條設定")
            self.resize(360, 210)

            layout = QtWidgets.QVBoxLayout(self)

            color_row = QtWidgets.QHBoxLayout()
            color_row.addWidget(QtWidgets.QLabel("顏色:"))
            self.color_btn = QtWidgets.QPushButton()
            self.color_btn.setFixedWidth(120)
            color_row.addWidget(self.color_btn)
            color_row.addStretch()
            layout.addLayout(color_row)

            style_row = QtWidgets.QHBoxLayout()
            style_row.addWidget(QtWidgets.QLabel("線型:"))
            self.style_combo = QtWidgets.QComboBox()
            for label, key in self.STYLE_OPTIONS:
                self.style_combo.addItem(label, key)
            style_row.addWidget(self.style_combo)
            style_row.addStretch()
            layout.addLayout(style_row)

            width_row = QtWidgets.QHBoxLayout()
            width_row.addWidget(QtWidgets.QLabel("粗度:"))
            self.width_spin = QtWidgets.QSpinBox()
            self.width_spin.setRange(1, 12)
            self.width_spin.setValue(int(width_value))
            width_row.addWidget(self.width_spin)
            width_row.addStretch()
            layout.addLayout(width_row)

            button_row = QtWidgets.QHBoxLayout()
            ok_btn = QtWidgets.QPushButton("套用")
            cancel_btn = QtWidgets.QPushButton("取消")
            button_row.addStretch()
            button_row.addWidget(ok_btn)
            button_row.addWidget(cancel_btn)
            layout.addLayout(button_row)

            ok_btn.clicked.connect(self.accept)
            cancel_btn.clicked.connect(self.reject)
            self.color_btn.clicked.connect(self.pick_color)

            self._color_rgb = tuple(int(v) for v in color_rgb)
            self._update_color_button()
            idx = self.style_combo.findData(style_name)
            self.style_combo.setCurrentIndex(idx if idx >= 0 else 0)

        def _update_color_button(self):
            r, g, b = self._color_rgb
            self.color_btn.setText(f"({r},{g},{b})")
            self.color_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #666;")

        def pick_color(self):
            color = get_tradingview_color()
            if color is not None and color.isValid():
                self._color_rgb = (color.red(), color.green(), color.blue())
                self._update_color_button()

        def get_values(self):
            style_name = str(self.style_combo.currentData())
            return self._color_rgb, style_name, int(self.width_spin.value())

    class LineToolManager:
        STYLE_TO_QT = {
            "solid": QtCore.Qt.SolidLine,
            "dashed": QtCore.Qt.DashLine,
            "dotted": QtCore.Qt.DotLine,
        }

        def __init__(self, target_plot, all_saved_lines, custom_pens):
            self.plot = target_plot
            self.saved_lines = all_saved_lines
            self.custom_pens = custom_pens
            self.line_configs = {}
            self.default_color = (0, 255, 255)
            self.default_style = "solid"
            self.default_width = 2

        def _make_pen(self, color_rgb, style_name, width_value):
            style = self.STYLE_TO_QT.get(style_name, QtCore.Qt.SolidLine)
            return pg.mkPen(color_rgb, width=max(int(width_value), 1), style=style)

        def _cfg_for_line(self, line_obj):
            return self.line_configs.get(
                line_obj,
                {
                    "color": tuple(self.default_color),
                    "style": self.default_style,
                    "width": int(self.default_width),
                },
            )

        def is_user_line(self, item_obj):
            return bool(getattr(item_obj, "_line_tool_tag", False))

        def register_line(self, line_obj, color_rgb=None, style_name=None, width_value=None):
            cfg = {
                "color": tuple(color_rgb if color_rgb is not None else self.default_color),
                "style": str(style_name if style_name is not None else self.default_style),
                "width": int(width_value if width_value is not None else self.default_width),
            }
            pen = self._make_pen(cfg["color"], cfg["style"], cfg["width"])
            line_obj.setPen(pen)
            try:
                line_obj._line_tool_tag = True
            except Exception:
                pass
            self.line_configs[line_obj] = cfg
            self.custom_pens[line_obj] = pen
            return pen

        def create_horizontal_line(self, y_val: float):
            x_range = self.plot.viewRange()[0]
            x_width = x_range[1] - x_range[0]
            x1 = x_range[0] + x_width * 0.2
            x2 = x_range[0] + x_width * 0.8
            line_obj = pg.LineSegmentROI([(x1, y_val), (x2, y_val)])
            line_obj.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
            self.plot.addItem(line_obj)
            self.saved_lines.append(line_obj)
            self.register_line(line_obj)
            return line_obj

        def create_segment_line(self, x1: float, y1: float, x2: float, y2: float):
            line_obj = pg.LineSegmentROI([(x1, y1), (x2, y2)])
            line_obj.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
            self.plot.addItem(line_obj)
            self.saved_lines.append(line_obj)
            self.register_line(line_obj)
            return line_obj

        def set_line_color(self, line_obj, color_rgb):
            if not self.is_user_line(line_obj):
                return
            cfg = self._cfg_for_line(line_obj)
            cfg["color"] = tuple(int(v) for v in color_rgb)
            self.register_line(line_obj, cfg["color"], cfg["style"], cfg["width"])

        def open_settings(self, target_line=None, parent=None):
            if target_line is None:
                base_color = self.default_color
                base_style = self.default_style
                base_width = self.default_width
            else:
                cfg = self._cfg_for_line(target_line)
                base_color = cfg["color"]
                base_style = cfg["style"]
                base_width = cfg["width"]

            dialog = LineSettingsDialog(base_color, base_style, base_width, parent)
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return False

            color_rgb, style_name, width_value = dialog.get_values()
            if target_line is None:
                self.default_color = tuple(color_rgb)
                self.default_style = style_name
                self.default_width = int(width_value)
                return True

            self.register_line(target_line, color_rgb, style_name, width_value)
            return True

        def unregister_line(self, line_obj):
            if line_obj in self.line_configs:
                del self.line_configs[line_obj]
            if line_obj in self.custom_pens:
                del self.custom_pens[line_obj]

        def clear_all(self):
            self.line_configs.clear()

    line_tool_manager = LineToolManager(plot, saved_lines, line_custom_pens)

    class FiboSettingsDialog(QtWidgets.QDialog):
        def __init__(self, multipliers, colors, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Fibo 設定")
            self.resize(420, 320)

            layout = QtWidgets.QVBoxLayout(self)
            tip = QtWidgets.QLabel("每列可設定一個倍數與對應顏色")
            layout.addWidget(tip)

            self.table = QtWidgets.QTableWidget(0, 2)
            self.table.setHorizontalHeaderLabels(["倍數", "顏色"])
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            layout.addWidget(self.table)

            control_row = QtWidgets.QHBoxLayout()
            add_btn = QtWidgets.QPushButton("新增列")
            remove_btn = QtWidgets.QPushButton("刪除列")
            control_row.addWidget(add_btn)
            control_row.addWidget(remove_btn)
            control_row.addStretch()
            layout.addLayout(control_row)

            button_row = QtWidgets.QHBoxLayout()
            ok_btn = QtWidgets.QPushButton("套用")
            cancel_btn = QtWidgets.QPushButton("取消")
            button_row.addStretch()
            button_row.addWidget(ok_btn)
            button_row.addWidget(cancel_btn)
            layout.addLayout(button_row)

            add_btn.clicked.connect(lambda: self.add_row(1.0, (255, 255, 255)))
            remove_btn.clicked.connect(self.remove_selected_row)
            ok_btn.clicked.connect(self.accept)
            cancel_btn.clicked.connect(self.reject)

            if not multipliers:
                multipliers = [0, 0.5, 1, 2, 3]
            if not colors:
                colors = [(255, 255, 255)] * len(multipliers)

            color_count = len(colors)
            for idx, mult in enumerate(multipliers):
                color = colors[idx % color_count] if color_count > 0 else (255, 255, 255)
                self.add_row(mult, color)

        @staticmethod
        def _normalize_rgb(color):
            return tuple(max(0, min(255, int(v))) for v in color)

        def _apply_color_button_style(self, button, color):
            r, g, b = self._normalize_rgb(color)
            button.setProperty("rgb", (r, g, b))
            button.setText(f"({r},{g},{b})")
            button.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #666;")

        def _pick_color_for_button(self, button):
            new_color = get_tradingview_color()
            if new_color is not None and new_color.isValid():
                self._apply_color_button_style(button, (new_color.red(), new_color.green(), new_color.blue()))

        def add_row(self, multiplier=1.0, color=(255, 255, 255)):
            row = self.table.rowCount()
            self.table.insertRow(row)

            mult_item = QtWidgets.QTableWidgetItem(f"{float(multiplier):g}")
            mult_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 0, mult_item)

            color_btn = QtWidgets.QPushButton()
            self._apply_color_button_style(color_btn, color)
            color_btn.clicked.connect(lambda _checked=False, b=color_btn: self._pick_color_for_button(b))
            self.table.setCellWidget(row, 1, color_btn)

        def remove_selected_row(self):
            row = self.table.currentRow()
            if row >= 0:
                self.table.removeRow(row)

        def get_values(self):
            multipliers = []
            colors = []
            for row in range(self.table.rowCount()):
                mult_item = self.table.item(row, 0)
                if mult_item is None:
                    continue
                text = mult_item.text().strip()
                if not text:
                    continue
                try:
                    mult = float(text)
                except Exception:
                    raise ValueError(f"第 {row + 1} 列倍數格式錯誤")

                color_btn = self.table.cellWidget(row, 1)
                rgb = color_btn.property("rgb") if color_btn is not None else (255, 255, 255)
                multipliers.append(mult)
                colors.append(self._normalize_rgb(rgb))

            if not multipliers:
                raise ValueError("請至少保留一列 Fibo 設定")

            return multipliers, colors

    class FiboManager:
        def __init__(self, target_plot, all_saved_lines, all_fibo_groups, custom_pens):
            self.plot = target_plot
            self.saved_lines = all_saved_lines
            self.fibo_groups = all_fibo_groups
            self.line_custom_pens = custom_pens
            self.default_multipliers = [0, 0.5, 1, 2, 3]
            self.default_colors = [
                (255, 255, 255),
                (255, 255, 255),
                (255, 255, 255),
                (255, 255, 255),
                (255, 255, 255),
            ]
            self.mode = False
            self.base_price = None

        def start_mode(self):
            self.mode = True
            self.base_price = None

        def cancel_mode(self):
            self.mode = False
            self.base_price = None

        def _normalize_config(self, multipliers, colors):
            m_list = [float(v) for v in multipliers] if multipliers else [0.0]
            c_list = [tuple(int(v) for v in color) for color in colors] if colors else [(255, 255, 255)]
            return m_list, c_list

        def _calc_line_ys(self, start_y: float, end_y: float, multipliers):
            delta = float(end_y - start_y)
            return [float(start_y + delta * mult) for mult in multipliers]

        def _calc_box_bounds(self, start_x: float, end_x: float, ys):
            x_left = float(min(start_x, end_x))
            x_right = float(max(start_x, end_x))
            if x_right <= x_left:
                x_right = x_left + 1e-6

            y_min = float(min(ys)) if ys else 0.0
            y_max = float(max(ys)) if ys else (y_min + 1e-6)
            if y_max <= y_min:
                y_max = y_min + 1e-6

            return x_left, x_right, y_min, y_max

        def _set_handle_line(self, group, handle_obj, x1: float, x2: float, y_val: float):
            handle_obj.blockSignals(True)
            try:
                left = float(min(x1, x2))
                width = max(float(abs(x2 - x1)), 1e-6)
                half_h = 0.5 * float(group.get("handle_height", 1e-6))
                handle_obj.setPos([left, float(y_val) - half_h])
                handle_obj.setSize([width, 2.0 * half_h])
            except Exception:
                pass
            handle_obj.blockSignals(False)

        def _get_tick_span(self) -> float:
            try:
                span = float(timeframe_to_seconds(timeframe))
                return span if span > 0 else 1.0
            except Exception:
                return 1.0

        def _get_handle_x_bounds(self, group):
            sx = float(group["start_point"][0])
            ex = float(group["end_point"][0])
            left = min(sx, ex)
            right = max(sx, ex)
            pad = 0.5 * self._get_tick_span()
            return left - pad, right + pad

        def get_group_by_item(self, item_obj):
            for group in self.fibo_groups:
                if item_obj in group.get("lines", []):
                    return group
                if item_obj is group.get("box"):
                    return group
                if item_obj is group.get("start_handle"):
                    return group
                if item_obj is group.get("end_handle"):
                    return group
            return None

        def get_group_by_line(self, line_obj):
            return self.get_group_by_item(line_obj)

        def handle_click(self, x_pos: float, y_pos: float):
            if not self.mode:
                return False

            if self.base_price is None:
                self.base_price = (float(x_pos), float(y_pos))
                return True

            self.create_group(
                start_point=self.base_price,
                end_point=(float(x_pos), float(y_pos)),
                multipliers=self.default_multipliers,
                colors=self.default_colors,
            )
            self.cancel_mode()
            return True

        def create_group(self, start_point, end_point, multipliers, colors):
            m_list, c_list = self._normalize_config(multipliers, colors)
            start_x, start_y = float(start_point[0]), float(start_point[1])
            end_x, end_y = float(end_point[0]), float(end_point[1])
            line_ys = self._calc_line_ys(start_y, end_y, m_list)
            x_left, x_right, y_min, y_max = self._calc_box_bounds(start_x, end_x, line_ys)

            # 群組控制框：可左右拉伸調整整組線長，也可整體平移
            box = pg.RectROI(
                [x_left, y_min],
                [x_right - x_left, y_max - y_min],
                pen=pg.mkPen((0, 0, 0, 0), width=1),
                movable=True,
            )
            box.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
            try:
                box.setPen(pg.mkPen((0, 0, 0, 0), width=1))
                box.setHoverPen(pg.mkPen((0, 0, 0, 0), width=1))
            except Exception:
                pass
            self.plot.addItem(box)

            # 0x 與 1x 控制線：薄矩形，可上下拖曳；每次同步時強制水平
            handle_x1 = x_left - 0.5 * self._get_tick_span()
            handle_x2 = x_right + 0.5 * self._get_tick_span()
            handle_h = max((y_max - y_min) * 0.003, 1e-6)
            start_handle = pg.RectROI(
                [handle_x1, start_y - 0.5 * handle_h],
                [max(handle_x2 - handle_x1, 1e-6), handle_h],
                pen=pg.mkPen((255, 80, 80), width=2),
                movable=True,
                rotatable=False,
                resizable=False,
            )
            end_handle = pg.RectROI(
                [handle_x1, end_y - 0.5 * handle_h],
                [max(handle_x2 - handle_x1, 1e-6), handle_h],
                pen=pg.mkPen((255, 80, 80), width=2),
                movable=True,
                rotatable=False,
                resizable=False,
            )
            start_handle.setZValue(9)
            end_handle.setZValue(9)
            start_handle.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
            end_handle.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
            self.plot.addItem(start_handle)
            self.plot.addItem(end_handle)

            group = {
                "lines": [],
                "box": box,
                "start_handle": start_handle,
                "end_handle": end_handle,
                "start_point": [start_x, start_y],
                "end_point": [end_x, end_y],
                "handle_height": handle_h,
                "multipliers": list(m_list),
                "colors": list(c_list),
                "last_box_rect": [x_left, x_right, y_min, y_max],
                "syncing": False,
            }
            self.fibo_groups.append(group)
            self.saved_lines.append(box)
            self.saved_lines.append(start_handle)
            self.saved_lines.append(end_handle)
            self._rebuild_lines(group)

            def _sync_box():
                self.sync_group(group)

            def _sync_handles():
                self.sync_group_by_handles(group)

            box.sigRegionChanged.connect(_sync_box)
            start_handle.sigRegionChanged.connect(_sync_handles)
            end_handle.sigRegionChanged.connect(_sync_handles)
            group["sync_callback_box"] = _sync_box
            group["sync_callback_handles"] = _sync_handles
            return group

        def _rebuild_lines(self, group):
            for ln in list(group.get("lines", [])):
                try:
                    self.plot.removeItem(ln)
                except Exception:
                    pass
                if ln in self.saved_lines:
                    self.saved_lines.remove(ln)
                if ln in self.line_custom_pens:
                    del self.line_custom_pens[ln]

            start_x, start_y = group["start_point"]
            end_x, end_y = group["end_point"]
            m_list = group["multipliers"]
            c_list = group["colors"]
            line_ys = self._calc_line_ys(start_y, end_y, m_list)

            lines = []
            for idx_mult, y_val in enumerate(line_ys):
                color = c_list[idx_mult % len(c_list)]
                line_pen = pg.mkPen(color, width=2)
                line_obj = pg.PlotDataItem(x=[start_x, end_x], y=[y_val, y_val], pen=line_pen)
                line_obj.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                self.plot.addItem(line_obj)
                lines.append(line_obj)
                self.saved_lines.append(line_obj)
                self.line_custom_pens[line_obj] = line_pen

            group["lines"] = lines
            self._update_box_geometry(group)

        def _update_box_geometry(self, group):
            start_x, start_y = group["start_point"]
            end_x, end_y = group["end_point"]
            ys = self._calc_line_ys(start_y, end_y, group["multipliers"])
            x_left, x_right, y_min, y_max = self._calc_box_bounds(start_x, end_x, ys)

            box = group.get("box")
            if box is not None:
                box.blockSignals(True)
                try:
                    box.setPos([x_left, y_min])
                    box.setSize([x_right - x_left, y_max - y_min])
                except Exception:
                    pass
                box.blockSignals(False)

            group["last_box_rect"] = [x_left, x_right, y_min, y_max]

        def sync_group(self, group):
            box = group.get("box")
            if box is None:
                return
            if group.get("syncing"):
                return

            group["syncing"] = True
            try:
                curr_pos = box.pos()
                curr_size = box.size()
                curr_left = float(curr_pos.x())
                curr_right = float(curr_pos.x() + curr_size.x())
                curr_ymin = float(curr_pos.y())
                curr_ymax = float(curr_pos.y() + curr_size.y())

                last_left, last_right, last_ymin, last_ymax = group.get(
                    "last_box_rect", [curr_left, curr_right, curr_ymin, curr_ymax]
                )

                if abs(curr_left - last_left) < 1e-12 and abs(curr_right - last_right) < 1e-12 and abs(curr_ymin - last_ymin) < 1e-12:
                    return

                start_is_left = group["start_point"][0] <= group["end_point"][0]
                if start_is_left:
                    group["start_point"][0] = curr_left
                    group["end_point"][0] = curr_right
                else:
                    group["start_point"][0] = curr_right
                    group["end_point"][0] = curr_left

                dy = curr_ymin - float(last_ymin)
                if abs(dy) > 1e-12:
                    group["start_point"][1] += dy
                    group["end_point"][1] += dy

                start_x, start_y = group["start_point"]
                end_x, end_y = group["end_point"]
                handle_x1, handle_x2 = self._get_handle_x_bounds(group)
                self._set_handle_line(group, group["start_handle"], handle_x1, handle_x2, start_y)
                self._set_handle_line(group, group["end_handle"], handle_x1, handle_x2, end_y)

                ys = self._calc_line_ys(start_y, end_y, group["multipliers"])
                for line_obj, y_val in zip(group.get("lines", []), ys):
                    try:
                        line_obj.setData([start_x, end_x], [y_val, y_val])
                    except Exception:
                        pass

                self._update_box_geometry(group)
            finally:
                group["syncing"] = False

        def sync_group_by_handles(self, group):
            if group.get("syncing"):
                return

            start_handle = group.get("start_handle")
            end_handle = group.get("end_handle")
            if start_handle is None or end_handle is None:
                return

            group["syncing"] = True
            try:
                s_pos = start_handle.pos()
                s_size = start_handle.size()
                e_pos = end_handle.pos()
                e_size = end_handle.size()
                start_y = float(s_pos.y() + 0.5 * s_size.y())
                end_y = float(e_pos.y() + 0.5 * e_size.y())

                start_x = float(group["start_point"][0])
                end_x = float(group["end_point"][0])
                group["start_point"][1] = start_y
                group["end_point"][1] = end_y

                handle_x1, handle_x2 = self._get_handle_x_bounds(group)
                self._set_handle_line(group, start_handle, handle_x1, handle_x2, start_y)
                self._set_handle_line(group, end_handle, handle_x1, handle_x2, end_y)

                ys = self._calc_line_ys(start_y, end_y, group["multipliers"])
                for line_obj, y_val in zip(group.get("lines", []), ys):
                    try:
                        line_obj.setData([start_x, end_x], [y_val, y_val])
                    except Exception:
                        pass

                self._update_box_geometry(group)
            finally:
                group["syncing"] = False

        def delete_group(self, group):
            box = group.get("box")
            if box is not None:
                try:
                    self.plot.removeItem(box)
                except Exception:
                    pass
                if box in self.saved_lines:
                    self.saved_lines.remove(box)
            start_handle = group.get("start_handle")
            if start_handle is not None:
                try:
                    self.plot.removeItem(start_handle)
                except Exception:
                    pass
                if start_handle in self.saved_lines:
                    self.saved_lines.remove(start_handle)
            end_handle = group.get("end_handle")
            if end_handle is not None:
                try:
                    self.plot.removeItem(end_handle)
                except Exception:
                    pass
                if end_handle in self.saved_lines:
                    self.saved_lines.remove(end_handle)
            for ln in list(group.get("lines", [])):
                try:
                    self.plot.removeItem(ln)
                except Exception:
                    pass
                if ln in self.saved_lines:
                    self.saved_lines.remove(ln)
                if ln in self.line_custom_pens:
                    del self.line_custom_pens[ln]
            if group in self.fibo_groups:
                self.fibo_groups.remove(group)

        def apply_group_settings(self, group, multipliers, colors):
            if not group:
                return False

            m_list, c_list = self._normalize_config(multipliers, colors)
            group["multipliers"] = list(m_list)
            group["colors"] = list(c_list)
            self._rebuild_lines(group)
            return True

        def open_group_settings(self, group, parent=None):
            if not group:
                return False

            dialog = FiboSettingsDialog(
                multipliers=group.get("multipliers", self.default_multipliers),
                colors=group.get("colors", self.default_colors),
                parent=parent,
            )
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return False

            try:
                multipliers, colors = dialog.get_values()
            except ValueError as e:
                QtWidgets.QMessageBox.warning(parent, "Fibo 設定", str(e))
                return False

            ok = self.apply_group_settings(group, multipliers, colors)
            if ok:
                self.default_multipliers = list(multipliers)
                self.default_colors = list(colors)
            return ok

        def get_tooltip_text(self):
            mult_text = ", ".join(f"{v:g}" for v in self.default_multipliers)
            return f"斐波那契水平線 (倍數: {mult_text})"

    fibo_manager = FiboManager(plot, saved_lines, fibo_groups, line_custom_pens)
    btn_fibo.setToolTip(fibo_manager.get_tooltip_text())

    # ================================================================
    def mouseClicked(evt):
        nonlocal line_mode, line_start, temp_line, selected_item, fibo_mode, fibo_base_price, text_mode, fibo_groups, line_custom_pens, range_mode, range_start_price, price_ranges

        pos = evt.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return

        point = plot.vb.mapSceneToView(pos)
        x, y = point.x(), point.y()

        # 檢查是否點擊了文字或線（若在 fibo/text/range 模式，略過現有線的選取以便建立新物件）
        clicked_item = None
        clicked_type = None  # "text" 或 "line"

        ignore_hit_test = fibo_mode or text_mode or range_mode

        # 先檢查文字，使用 boundingRect
        if not ignore_hit_test:
            for item in saved_lines:
                if not item.isVisible():
                    continue
                if isinstance(item, pg.TextItem):
                    local_pos = item.mapFromScene(pos)
                    if item.boundingRect().contains(local_pos):
                        clicked_item = item
                        clicked_type = "text"
                        break

        # 再檢查劃線
        if not ignore_hit_test and clicked_item is None:
            for item in saved_lines:
                if not item.isVisible() or isinstance(item, pg.TextItem):
                    continue
                if isinstance(item, (pg.RectROI, pg.CircleROI)):
                    try:
                        local_pos = item.mapFromScene(pos)
                        if item.boundingRect().contains(local_pos):
                            clicked_item = item
                            clicked_type = "line"
                            break
                    except Exception:
                        continue
                if isinstance(item, pg.PlotDataItem):
                    try:
                        x_data, y_data = item.getData()
                        if x_data is None or y_data is None or len(x_data) < 2:
                            continue
                        x1, y1 = float(x_data[0]), float(y_data[0])
                        x2, y2 = float(x_data[-1]), float(y_data[-1])

                        if abs(x2 - x1) < 0.1:
                            dist = abs(x - x1)
                        elif abs(y2 - y1) < 0.1:
                            dist = abs(y - y1)
                        else:
                            num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
                            den = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
                            dist = num / den if den != 0 else float('inf')

                        x_min, x_max = min(x1, x2), max(x1, x2)
                        y_min, y_max = min(y1, y2), max(y1, y2)
                        if x_min - 10 <= x <= x_max + 10 and y_min - 10 <= y <= y_max + 10 and dist <= 10:
                            clicked_item = item
                            clicked_type = "line"
                            break
                    except Exception:
                        continue
                if isinstance(item, pg.InfiniteLine):
                    try:
                        if int(getattr(item, "angle", 0)) == 0:
                            y_line = float(item.value())
                            x_min = float(plot.viewRange()[0][0])
                            x_max = float(plot.viewRange()[0][1])
                            if x_min > x_max:
                                x_min, x_max = x_max, x_min
                            if x_min <= x <= x_max and abs(y - y_line) <= 10:
                                clicked_item = item
                                clicked_type = "line"
                                break
                    except Exception:
                        continue
                try:
                    line_start_pos = item.getState()['points'][0]
                    line_end_pos = item.getState()['points'][1]

                    x1, y1 = line_start_pos[0], line_start_pos[1]
                    x2, y2 = line_end_pos[0], line_end_pos[1]

                    if abs(x2 - x1) < 0.1:
                        dist = abs(x - x1)
                    elif abs(y2 - y1) < 0.1:
                        dist = abs(y - y1)
                    else:
                        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
                        den = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
                        dist = num / den if den != 0 else float('inf')

                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    if x_min - 10 <= x <= x_max + 10 and y_min - 10 <= y <= y_max + 10 and dist <= 10:
                        clicked_item = item
                        clicked_type = "line"
                        break
                except Exception:
                    continue

        # 右鍵：文字或線的上下文菜單
        if evt.button() == QtCore.Qt.RightButton:
            if clicked_item:
                if clicked_type == "text":
                    target_text = clicked_item
                    menu = QtWidgets.QMenu()
                    del_act = menu.addAction("刪除")
                    color_act = menu.addAction("改顏色")
                    size_act = menu.addAction("改大小")

                    def delete_text():
                        nonlocal selected_item
                        try:
                            plot.removeItem(target_text)
                        except Exception:
                            pass
                        if target_text in saved_lines:
                            saved_lines.remove(target_text)
                        if selected_item == target_text:
                            selected_item = None

                    def change_text_color():
                        color = get_tradingview_color()
                        if color and color.isValid():
                            target_text.setColor(color)

                    def change_text_size():
                        font = QtGui.QFont()
                        if hasattr(target_text, "textItem"):
                            font = target_text.textItem.font()
                        current_size = font.pointSize() if font.pointSize() > 0 else 12
                        new_size, ok = QtWidgets.QInputDialog.getInt(None, "字體大小", "字體大小 (pt):", current_size, 6, 64)
                        if ok:
                            font.setPointSize(int(new_size))
                            try:
                                target_text.textItem.setFont(font)
                            except Exception:
                                target_text.setFont(font)

                    del_act.triggered.connect(delete_text)
                    color_act.triggered.connect(change_text_color)
                    size_act.triggered.connect(change_text_size)
                    menu.exec(QtGui.QCursor.pos())
                else:
                    target_line = clicked_item

                    target_group = fibo_manager.get_group_by_item(target_line)
                    is_user_line = line_tool_manager.is_user_line(target_line)

                    context_menu = QtWidgets.QMenu()
                    settings_action = context_menu.addAction("Fibo 設定") if target_group else None
                    line_settings_action = context_menu.addAction("線條設定") if (not target_group and is_user_line) else None
                    delete_action = context_menu.addAction("刪除")
                    color_action = context_menu.addAction("改顏色")

                    def open_fibo_settings():
                        nonlocal selected_item
                        if not target_group:
                            return
                        if selected_item == target_group.get("box") or selected_item == target_group.get("start_handle") or selected_item == target_group.get("end_handle") or selected_item in target_group.get("lines", []):
                            selected_item = None
                        if fibo_manager.open_group_settings(target_group, None):
                            btn_fibo.setToolTip(fibo_manager.get_tooltip_text())

                    def delete_line():
                        nonlocal selected_item
                        # 检查是否为价格范围测量线
                        target_range = None
                        if isinstance(target_line, pg.LineSegmentROI):
                            for rg in price_ranges:
                                if rg.get('line') == target_line:
                                    target_range = rg
                                    break
                        
                        if target_range:
                            # 删除测量线和标签
                            try:
                                plot.removeItem(target_range['line'])
                            except Exception:
                                pass
                            try:
                                plot.removeItem(target_range['label'])
                            except Exception:
                                pass
                            
                            # 从saved_lines中移除
                            if target_range['line'] in saved_lines:
                                saved_lines.remove(target_range['line'])
                            if target_range['label'] in saved_lines:
                                saved_lines.remove(target_range['label'])
                            
                            price_ranges.remove(target_range)
                            if selected_item == target_range['line']:
                                selected_item = None
                        elif target_group:
                            fibo_manager.delete_group(target_group)
                            if selected_item == target_group.get("box") or selected_item == target_group.get("start_handle") or selected_item == target_group.get("end_handle") or selected_item in target_group["lines"]:
                                selected_item = None
                        else:
                            try:
                                plot.removeItem(target_line)
                            except Exception:
                                pass
                            if target_line in saved_lines:
                                saved_lines.remove(target_line)
                            line_tool_manager.unregister_line(target_line)
                            if selected_item == target_line:
                                selected_item = None

                    def open_line_settings():
                        if line_tool_manager.open_settings(target_line, None):
                            pass

                    def change_color():
                        nonlocal line_custom_pens
                        color = get_tradingview_color()
                        if color and color.isValid():
                            r, g, b = color.red(), color.green(), color.blue()
                            pen = pg.mkPen((r, g, b), width=2)
                            if target_group:
                                applied_colors = [(r, g, b)] * len(target_group.get("multipliers", []))
                                fibo_manager.apply_group_settings(
                                    target_group,
                                    target_group.get("multipliers", [0.0]),
                                    applied_colors,
                                )
                            elif is_user_line:
                                line_tool_manager.set_line_color(target_line, (r, g, b))
                            else:
                                target_line.setPen(pen)
                                line_custom_pens[target_line] = pen

                    if settings_action:
                        settings_action.triggered.connect(open_fibo_settings)
                    if line_settings_action:
                        line_settings_action.triggered.connect(open_line_settings)
                    delete_action.triggered.connect(delete_line)
                    color_action.triggered.connect(change_color)
                    context_menu.exec(QtGui.QCursor.pos())
            else:
                blank_menu = QtWidgets.QMenu()
                line_default_settings_action = blank_menu.addAction("劃線設定")

                def open_default_line_settings():
                    line_tool_manager.open_settings(None, None)

                line_default_settings_action.triggered.connect(open_default_line_settings)
                blank_menu.exec(QtGui.QCursor.pos())
            return

        # 左鍵點擊邏輯
        if evt.button() != QtCore.Qt.LeftButton:
            return

        if clicked_item:
            if clicked_type == "line":
                # 切換選中狀態並高亮線條
                if isinstance(selected_item, pg.LineSegmentROI) and selected_item is not clicked_item:
                    # 恢復上一個選中線的原始顏色
                    original_pen = line_custom_pens.get(selected_item, pg.mkPen("cyan", width=2))
                    selected_item.setPen(original_pen)
                if selected_item == clicked_item:
                    # 取消選中，恢復原始顏色
                    original_pen = line_custom_pens.get(selected_item, pg.mkPen("cyan", width=2))
                    selected_item.setPen(original_pen)
                    selected_item = None
                else:
                    selected_item = clicked_item
                    clicked_item.setPen(pg.mkPen("yellow", width=3))
            else:
                # 選中文字，保留現有高亮
                if isinstance(selected_item, pg.LineSegmentROI):
                    original_pen = line_custom_pens.get(selected_item, pg.mkPen("cyan", width=2))
                    selected_item.setPen(original_pen)
                selected_item = clicked_item
            return

        if not line_mode and not fibo_mode and not text_mode and not range_mode:
            return
        
        # 處理價格範圍測量模式
        if range_mode:
            if range_start_price is None:
                # 第一次點擊：記錄起點
                range_start_price = y
                return
            else:
                # 第二次點擊：創建一條可調整端點的測量線
                y1 = range_start_price
                y2 = y
                
                # 確保 y1 < y2
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # 使用點擊位置的X座標
                x_pos = x
                
                # 創建一條垂直線，端點可拖動
                measure_line = pg.LineSegmentROI(
                    [(x_pos, y1), (x_pos, y2)],
                    pen=pg.mkPen((100, 200, 255), width=2)
                )
                measure_line.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.RightButton)
                
                plot.addItem(measure_line)
                
                # 計算價格差異
                price_diff = y2 - y1
                price_pct = (price_diff / y1) * 100 if y1 != 0 else 0
                
                # 創建文字標籤顯示價格資訊（放在線的右側中間）
                label_text = f"{price_diff:.3f} ({price_pct:.3f}%)"
                text_label = pg.TextItem(text=label_text, color=(100, 200, 255), anchor=(0, 0.5))
                x_range = plot.viewRange()[0]
                x_width = x_range[1] - x_range[0]
                text_label.setPos(x_pos + x_width * 0.02, (y1 + y2) / 2)
                plot.addItem(text_label)
                
                # 保存範圍物件（作為群組）
                range_group = {
                    'line': measure_line,
                    'label': text_label
                }
                price_ranges.append(range_group)
                saved_lines.append(measure_line)
                saved_lines.append(text_label)
                
                # 添加拖曳時更新標籤的回調
                def update_range_display():
                    # 獲取線的當前端點位置
                    points = measure_line.getState()['points']
                    
                    y_bottom = points[0][1]
                    y_top = points[1][1]
                    x_line = points[0][0]
                    
                    # 確保 y_bottom < y_top
                    if y_bottom > y_top:
                        y_bottom, y_top = y_top, y_bottom
                    
                    # 更新價格差異
                    new_diff = abs(y_top - y_bottom)
                    new_pct = (new_diff / y_bottom) * 100 if y_bottom != 0 else 0
                    text_label.setText(f"{new_diff:.3f} ({new_pct:.3f}%)")
                    
                    # 更新標籤位置（放在線的右側中間）
                    x_range = plot.viewRange()[0]
                    x_width = x_range[1] - x_range[0]
                    label_x = x_line + x_width * 0.02
                    text_label.setPos(label_x, (y_bottom + y_top) / 2)
                
                measure_line.sigRegionChanged.connect(update_range_display)
                
                range_mode = False
                range_start_price = None
                return
        
        # 處理 Fibo 模式（群組，同步調整）
        if fibo_mode:
            if fibo_manager.handle_click(float(x), float(y)):
                fibo_mode = fibo_manager.mode
                fibo_base_price = fibo_manager.base_price
                return
        
        # 處理 Text 模式
        if text_mode:
            dialog = TextInputDialog()
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                text, color_rgb, font_size = dialog.get_values()
                if text:
                    # 在點擊位置添加文字
                    text_item = pg.TextItem(text=text, color=color_rgb, anchor=(0, 0))
                    text_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
                    # 左鍵拖曳、右鍵穿透到場景以觸發自訂選單
                    text_item.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
                    text_item.setZValue(5)
                    
                    # 設定字體大小
                    font = QtGui.QFont()
                    font.setPointSize(font_size)
                    try:
                        text_item.textItem.setFont(font)
                    except Exception:
                        pass
                    
                    text_item.setPos(x, y)
                    plot.addItem(text_item)
                    saved_lines.append(text_item)  # 保存文字項目
            text_mode = False
            return
        
        if not line_mode:
            return
        
        # 處理劃線模式
        if line_mode == "horizontal":
            line_tool_manager.create_horizontal_line(float(y))
            line_mode = False
            return
        
        # 處理普通劃線模式
        if line_start is None:
            line_start = (x, y)
            return

        x1, y1 = line_start
        line_tool_manager.create_segment_line(float(x1), float(y1), float(x), float(y))

        if temp_line:
            plot.removeItem(temp_line)
            temp_line = None

        line_start = None
        line_mode = False

    plot.scene().sigMouseClicked.connect(mouseClicked)

    # ================================================================
    # update（下一根）
    # ================================================================
    def update():
        nonlocal idx
        if idx >= len(df):
            return

        def set_scatter_by_mask(scatter_obj, x_vals, y_vals, mask_vals):
            x_arr = np.asarray(x_vals, dtype=float)
            y_arr = np.asarray(y_vals, dtype=float)
            mask_arr = np.asarray(mask_vals, dtype=bool)
            if x_arr.size == 0 or y_arr.size == 0 or not mask_arr.any():
                scatter_obj.setData(x=[], y=[])
                return
            scatter_obj.setData(x=x_arr[mask_arr], y=y_arr[mask_arr])

        bar = df.iloc[idx]
        draw_bar(bar)

        # 新K出現後，處理委託
        process_pending_orders(bar)

        ma_vals = df.iloc[replay_start_idx:idx+1]["MA20"].values
        ma_vals_100 = df.iloc[replay_start_idx:idx+1]["MA100"].values
        times = df.iloc[replay_start_idx:idx+1]["dt_based_ts"].values
        ma_curve_20.setData(times, ma_vals)
        if timeframe == "1m":
            ma_curve_100.setData(times, ma_vals_100)
        bull_mask = (df.iloc[replay_start_idx:idx+1]["close"] > df.iloc[replay_start_idx:idx+1]["open"]).values
        bear_mask = (df.iloc[replay_start_idx:idx+1]["close"] < df.iloc[replay_start_idx:idx+1]["open"]).values
        set_scatter_by_mask(
            bull_bar_scatter,
            times,
            df.iloc[replay_start_idx:idx+1]["bull_bar_score"].values,
            bull_mask,
        )
        set_scatter_by_mask(
            bear_bar_scatter,
            times,
            df.iloc[replay_start_idx:idx+1]["bear_bar_score"].values,
            bear_mask,
        )
        bull_step_x, bull_step_y = centered_step_xy(times, df.iloc[replay_start_idx:idx+1]["bull_run"].values)
        bear_step_x, bear_step_y = centered_step_xy(times, df.iloc[replay_start_idx:idx+1]["bear_run"].values)
        bull_run_curve.setData(bull_step_x, bull_step_y)
        bear_run_curve.setData(bear_step_x, bear_step_y)
        update_indicator_axis_range(idx)

        offset = (bar["high"] - bar["low"]) * 0.3
        txt = pg.TextItem(
            text=str(idx - replay_start_idx + 1),
            color="white",
            anchor=(0.5, 1.0)
        )
        txt.setPos(bar["dt_based"].timestamp(), bar["low"] - offset)
        plot.addItem(txt)
        draw_vertical_grids()

        update_indicator_value_panel(idx)
        update_kline_info_panel(idx)

        idx += 1
        if auto_all_mode:
            apply_auto_all_view()
        update_pnl_labels()
        refresh_visible_candle_record()

    # ================================================================
    # key press
    # ================================================================
    def keyPress(evt):
        nonlocal idx, line_mode, line_start, temp_line, selected_item, fibo_groups, line_custom_pens
        key = evt.key()

        if key == QtCore.Qt.Key_Delete:
            if selected_item is not None:
                target_group = None
                if isinstance(selected_item, pg.LineSegmentROI):
                    for g in fibo_groups:
                        if selected_item in g.get("lines", []):
                            target_group = g
                            break
                if target_group:
                    for ln in list(target_group["lines"]):
                        if ln in saved_lines:
                            try:
                                plot.removeItem(ln)
                            except Exception:
                                pass
                            saved_lines.remove(ln)
                            if ln in line_custom_pens:
                                del line_custom_pens[ln]
                    fibo_groups.remove(target_group)
                else:
                    try:
                        plot.removeItem(selected_item)
                    except Exception:
                        pass
                    if selected_item in saved_lines:
                        saved_lines.remove(selected_item)
                    line_tool_manager.unregister_line(selected_item)
                selected_item = None
            return

        if key == QtCore.Qt.Key_Right:
            update()

        elif key == QtCore.Qt.Key_Left:
            if idx > replay_start_idx:
                idx -= 1
                redraw_all()

    # 將鍵盤事件綁定在頂層視窗
    root.keyPressEvent = keyPress

    # 使用全域快捷鍵，避免焦點在子元件時左右鍵失效
    shortcut_right = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), root)
    shortcut_right.setContext(QtCore.Qt.ApplicationShortcut)
    shortcut_right.activated.connect(update)

    def step_back():
        nonlocal idx
        if idx > replay_start_idx:
            idx -= 1
            redraw_all()

    shortcut_left = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), root)
    shortcut_left.setContext(QtCore.Qt.ApplicationShortcut)
    shortcut_left.activated.connect(step_back)

    def on_close(event):
        QtWidgets.QApplication.quit()

    root.closeEvent = on_close
    root.setFocusPolicy(QtCore.Qt.StrongFocus)
    root.setFocus()

    if len(df) > 0:
        hline.setPos(float(df.iloc[replay_start_idx]["close"]))
        hline_indicator.setPos(0.0)
        set_crosshair_visibility("none")
        update_kline_info_panel(replay_start_idx)
        update_axis_value_boxes(replay_start_idx)

    root.resize(1400, 800)
    root.show()
    app.exec()


# ================================================================
# 程式進入點
# ================================================================
if __name__ == "__main__":
    main()
