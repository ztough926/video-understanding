#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_keyframes.py —— 从视频 / GIF 中抽取「关键帧」

关键帧的定义：画面发生实质变化的那一帧。
具体做法是逐帧比较相邻两帧的画面差异，只有当差异超过阈值（画面真的变了）时才
把它选出来；停着不动的那些重复帧一律丢掉。

输出（默认）：
  <output>/overview.jpg               关键帧总览图（帧数多时自动分成 overview_01.jpg ...
                                      overview_NN.jpg 多张，**每张都保证每格看得清**）。
                                      图面上只有画面网格 + 每格下方的 #序号/时间戳，
                                      不印元数据条（尺寸/时长/帧率见 keyframes.json）
  <output>/frames/kf_001_0.000s.jpg   按时间顺序编号的单帧图片
  <output>/keyframes.json             元数据：原始尺寸 / 总时长 / 帧率 / 每帧时间戳

输出（--sheet-only，只出总览图）：
  <视频同目录>/<文件名>_overview.jpg（多张时 _overview_01.jpg ...）

依赖：ffmpeg + ffprobe（系统命令）、numpy、Pillow（总览图 / GIF 需要）

用法：
  python extract_keyframes.py video.mp4 --sheet-only
  python extract_keyframes.py video.mp4 -o out --threshold 0.03 --max-width 960
  python extract_keyframes.py a.mp4 b.gif --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


# --------------------------------------------------------------------- 默认参数
ANALYZE_WIDTH = 200      # 分析用的缩略图宽度（越小越快，200 足够判断画面变化）
BLOCK_GRID = 8           # 局部差异检测的分块网格（8x8），用来抓画面局部的小变化
DEFAULT_NOISE_K = 4.0    # 自适应阈值 = 噪声底 x K
DEFAULT_ABS_FLOOR = 0.008  # 自适应阈值下限，避免静止视频把噪声当变化
DEFAULT_ABS_CEIL = 0.080   # 自适应阈值上限，避免噪声大的视频什么都抓不到
SHEET_WIDTH = 9600        # 总览图目标宽度（像素），越大每格越清晰
SHEET_MAX_CELL_WIDTH = 1600   # 单格宽度上限，防止 4K 源把总图撑到离谱
# 总览图最多放多少格，超了按时间抽稀。这是"信息量 vs 每格清晰度"的旋钮：
# 大模型看整张图时会按自己的像素预算把它缩小，格子越多每格分到的像素越少。
# 实测同一段 3 分钟视频，4800px 宽的图：64 格时每格内容糊、只能看出版式，
# 32 格时每格里的标题已经能读出来。所以默认取 32，宁可少几格也别糊。
SHEET_MAX_CELLS = 32
# 一张图最多放多少格（硬上限）。读图工具会把整张图压到长边约 1568px 再交给模型，
# 所以每格能分到的像素只取决于"这张图里有多少格"。一张塞 40 格时每格只剩 ~250px、
# 小字全糊；12~16 格时每格能拿到 350~390px，字幕和标题都读得出来。超过这个上限的
# 帧不要往同一张里塞，**分到下一张去**——清晰度是第一位的。
SHEET_PER_SHEET_CAP = 16
# 最多生成几张总览图。帧更多时先抽稀再分张，不再无限加张。
SHEET_MAX_SHEETS = 4

_VIDEO_EXT = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v",
    ".mpg", ".mpeg", ".ts", ".m2ts", ".3gp", ".ogv", ".rmvb", ".vob",
}
_GIF_EXT = {".gif"}

# Windows 下不要弹出控制台窗口
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# --------------------------------------------------------------------- 小工具
def _which(name: str) -> str:
    p = shutil.which(name)
    if p:
        return p
    # 常见安装位置兜底
    candidates = [
        rf"C:\ffmpeg\bin\{name}.exe",
        rf"D:\ffmpeg\bin\{name}.exe",
        rf"D:\ffmpeg-8.1.2-full_build\bin\{name}.exe",
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise RuntimeError(
        f"找不到 {name}，请先安装 ffmpeg 并加入 PATH（https://ffmpeg.org/download.html）"
    )


def _run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd), capture_output=True, check=False, creationflags=_NO_WINDOW, **kw
    )


def fmt_time(seconds: float) -> str:
    """秒 -> 便于读的时间串"""
    if seconds is None or seconds != seconds:
        return "?"
    m, s = divmod(float(seconds), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}" if h else f"{int(m):d}:{s:06.3f}"


def _safe_float(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default


def _even(n: int) -> int:
    n = int(n)
    return n if n % 2 == 0 else max(2, n - 1)


# --------------------------------------------------------------------- 元数据
def probe(path: Path) -> dict:
    """用 ffprobe 读取视频的原始尺寸 / 帧率 / 时长 / 总帧数"""
    ffprobe = _which("ffprobe")
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name",
        "-show_entries", "stream_side_data=rotation",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    res = _run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe 读取失败：{path}\n{res.stderr.decode('utf-8', 'replace')}")
    data = json.loads(res.stdout.decode("utf-8", "replace") or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"文件里没有视频流：{path}")
    st = streams[0]

    width = int(st.get("width") or 0)
    height = int(st.get("height") or 0)

    # 手机竖拍视频带旋转元数据，ffmpeg 解码时自动摆正，这里同步校正"看到的尺寸"
    rotation = 0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(_safe_float(sd.get("rotation"), 0) or 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width

    # 帧率：avg_frame_rate 更接近实际，退化时用 r_frame_rate
    def parse_rate(v):
        if not v or "/" not in str(v):
            return _safe_float(v, 0.0) or 0.0
        num, den = str(v).split("/")[:2]
        num, den = _safe_float(num, 0.0) or 0.0, _safe_float(den, 0.0) or 0.0
        return num / den if den else 0.0

    fps = parse_rate(st.get("avg_frame_rate")) or parse_rate(st.get("r_frame_rate")) or 0.0

    duration = _safe_float(st.get("duration"))
    if not duration:
        duration = _safe_float((data.get("format") or {}).get("duration"))

    nb_frames = int(_safe_float(st.get("nb_frames"), 0) or 0)
    if not duration and nb_frames and fps:
        duration = nb_frames / fps
    if not nb_frames and duration and fps:
        nb_frames = int(round(duration * fps))

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "duration": round(duration, 4) if duration else None,
        "nb_frames": nb_frames or None,
        "codec": st.get("codec_name"),
        "rotation": rotation,
    }


# --------------------------------------------------------------------- 帧差计算
def diff_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    两帧（已灰度归一化的小图，取值 0~1）之间的差异分数。

    取两个角度的较大者：
      · 全局平均差异 —— 整幅画面的整体变化（转场、整屏换色）
      · 局部块差异的最大值 —— 画面某一小块的变化（角落出字幕、局部动效）

    为什么要看局部块：画面里一个小物体移动时，它只占很小一块面积，算全局平均值
    会被大量没变化的像素稀释掉（实测只有 0.004，几乎等于噪声）。把画面切成 8x8
    块、只看"变化最剧烈的那一块"，同一个动作能拿到 0.06 左右，区分度好得多。
    每块内部取均值本身就能抵消随机噪点，所以取最大值也不会被噪声带偏。

    值域 0~1，0 表示两帧完全一样。
    """
    d = np.abs(a - b)
    global_mae = float(d.mean())

    h, w = d.shape
    bh, bw = h // BLOCK_GRID, w // BLOCK_GRID
    if bh >= 2 and bw >= 2:
        blocks = d[: bh * BLOCK_GRID, : bw * BLOCK_GRID].reshape(
            BLOCK_GRID, bh, BLOCK_GRID, bw
        ).mean(axis=(1, 3))
        local = float(blocks.max())
    else:
        local = global_mae

    return max(global_mae, local)


def compute_threshold(
    scores: np.ndarray,
    manual: float = 0.0,
    noise_k: float = DEFAULT_NOISE_K,
    floor: float = DEFAULT_ABS_FLOOR,
    ceil: float = DEFAULT_ABS_CEIL,
) -> tuple[float, str]:
    """
    判定"画面变了"的阈值。

    自动模式：视频里大部分时间静止，所以相邻帧差异的中位数约等于"底噪"
    （压缩噪点 / 轻微抖动）。把阈值取成底噪的 K 倍，并夹在 [floor, ceil] 之间。
    这样既不会被噪声骗，也不会因为视频整体噪点大而一个变化都抓不到。
    """
    if manual and manual > 0:
        return float(manual), "manual"
    if scores.size == 0:
        return floor, "auto"
    noise = float(np.median(scores))
    t = min(max(noise * noise_k, floor), ceil)
    return t, "auto"


def select_keyframes(
    scores: np.ndarray,
    total_frames: int,
    fps: float,
    threshold: float,
    min_time: float = 0.30,
    motion_time: float = 1.5,
    max_frames: int = 200,
    include_last: bool = False,
) -> list[int]:
    """
    挑出关键帧的帧号，返回的时间顺序即原视频顺序。

    规则：
      1. 第一帧一定保留（初始状态）。
      2. 画面一直在变（diff > 阈值）时不急着选，等它变完 —— 变化结束后那个
         不再变化的帧才是"完整展示"的画面，选它。
      3. 如果画面一直变个不停（运镜 / 长动效），每 motion_time 秒强制补一帧，
         免得一整个运动段只出一张图。
      4. 相邻关键帧至少间隔 min_time 秒，防止瞬间抖动炸出一堆重复帧。
    """
    if total_frames <= 0:
        return []
    # 帧数本身就极少（短 GIF）：每一帧都是独立内容，全都留下
    if total_frames <= 8:
        return list(range(total_frames))

    selected: list[int] = [0]
    min_gap = max(1, int(round(min_time * fps))) if fps and fps > 0 else 1
    motion_gap = max(1, int(round(motion_time * fps))) if fps and fps > 0 else 1

    in_change = False
    change_start = 0
    for i, s in enumerate(scores, start=1):
        if s > threshold:
            if not in_change:
                in_change = True
                change_start = i
            # 变了很久还没停下来（运镜 / 长动效）：每 motion_gap 帧补一张，
            # 不然一整段运动只会得到一张图
            if i - change_start >= motion_gap and i - selected[-1] >= min_gap:
                selected.append(i)
                change_start = i
        elif in_change:
            # 变化刚结束，这一帧画面已经更新完且稳定 —— 就是它
            if i - selected[-1] >= min_gap:
                selected.append(i)
            in_change = False

    last = total_frames - 1
    # 到片尾还在变（最后一个镜头没停下来）：补上结尾画面，不然会丢掉最后一个状态
    if in_change and last > selected[-1] and last - selected[-1] >= min_gap:
        selected.append(last)
    if include_last and last > selected[-1] and last - selected[-1] >= min_gap:
        selected.append(last)

    if max_frames and len(selected) > max_frames:
        pick = np.linspace(0, len(selected) - 1, int(max_frames)).round().astype(int)
        selected = [selected[i] for i in sorted(set(pick.tolist()))]

    return selected


# --------------------------------------------------------------------- 第一遍：分析
def _iter_video_gray(path: Path, aw: int, ah: int) -> Iterator[np.ndarray]:
    """让 ffmpeg 直接输出 aw x ah 的灰度裸帧流，逐帧 yield（uint8 二维数组）"""
    ffmpeg = _which("ffmpeg")
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
        "-vf", f"scale={aw}:{ah}:flags=area",
        "-pix_fmt", "gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-",
    ]
    nbytes = aw * ah
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=errf,
            creationflags=_NO_WINDOW,
        )
        assert proc.stdout is not None
        try:
            while True:
                buf = proc.stdout.read(nbytes)
                if not buf or len(buf) < nbytes:
                    break
                yield np.frombuffer(buf, dtype=np.uint8).reshape(ah, aw)
        finally:
            proc.stdout.close()
            proc.wait()
            errf.seek(0)
            err = errf.read().decode("utf-8", "replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 解码失败：{path}\n{err}")


def _iter_gif_frames(path: Path) -> Iterator[np.ndarray]:
    """逐帧读出 GIF（Pillow 已处理帧间 disposal，拿到的是完整画面）"""
    if not _HAS_PIL:
        raise RuntimeError("处理 GIF 需要 Pillow：pip install Pillow")
    with Image.open(path) as im:
        for i in range(getattr(im, "n_frames", 1)):
            im.seek(i)
            yield np.asarray(im.convert("RGB"))


def _downscale_gray(rgb: np.ndarray, aw: int) -> np.ndarray:
    """RGB 数组 -> aw 宽左右的灰度缩略图（块平均降采样，快且抗噪）"""
    h, w = rgb.shape[:2]
    ah = max(2, int(round(h * aw / w)))
    # 用 Pillow 走一次高质量缩放（GIF 帧通常不大，开销可忽略）
    img = Image.fromarray(rgb).convert("L").resize((aw, ah), Image.BOX)
    return np.asarray(img)


def analyze(path: Path, analyze_width: int = ANALYZE_WIDTH):
    """第一遍：逐帧算相邻帧差异，返回 (差异数组, 实际帧数, 元数据)"""
    suffix = path.suffix.lower()
    is_gif = suffix in _GIF_EXT

    if is_gif:
        info = {"width": 0, "height": 0, "fps": 0.0, "duration": None,
                "nb_frames": None, "codec": "gif", "rotation": 0}
        if _HAS_PIL:
            with Image.open(path) as im:
                info["width"], info["height"] = im.size
                n = getattr(im, "n_frames", 1)
                info["nb_frames"] = n
                total_ms = 0.0
                for i in range(n):
                    im.seek(i)
                    total_ms += float(im.info.get("duration") or 0)
                info["duration"] = round(total_ms / 1000.0, 4) if total_ms else None
                info["fps"] = round(1000.0 / (total_ms / n), 4) if total_ms and n else 0.0
        source = _iter_gif_frames(path)
    else:
        info = probe(path)
        aw = min(analyze_width, info["width"] or analyze_width)
        aw = _even(aw)
        ah = _even(max(2, int(round((info["height"] or aw) * aw / (info["width"] or aw)))))
        source = _iter_video_gray(path, aw, ah)

    scores: list[float] = []
    count = 0
    prev = None
    for frame in source:
        if frame.ndim == 3:
            frame = _downscale_gray(frame, min(analyze_width, frame.shape[1]))
        sig = frame.astype(np.float32) / 255.0
        if prev is not None:
            scores.append(diff_score(prev, sig))
        prev = sig
        count += 1

    return np.asarray(scores, dtype=np.float32), count, info


# --------------------------------------------------------------------- 第二遍：导出
def _export_video(path: Path, indices: Sequence[int], out_size: tuple[int, int],
                  out_files: Sequence[Path], quality: int) -> None:
    """
    第二遍：完整解码一遍，遇到指定的帧号就存图。

    这里没有用 ffmpeg 的 select 滤镜（`select='eq(n,0)+eq(n,17)+...'`）——
    关键帧上百个以后那个表达式会超出 ffmpeg 解析器的上限而直接报错，
    所以改成让 ffmpeg 吐裸帧流、在 Python 侧按帧号筛选，解一遍就够。
    """
    if not _HAS_PIL:
        raise RuntimeError("导出关键帧需要 Pillow：pip install Pillow")

    ffmpeg = _which("ffmpeg")
    out_w, out_h = out_size
    frame_bytes = out_w * out_h * 3
    wanted = {int(i): k for k, i in enumerate(indices)}   # 帧号 -> 第几张

    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
        "-vf", f"scale={out_w}:{out_h}:flags=lanczos",
        "-pix_fmt", "rgb24", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-",
    ]
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=errf, creationflags=_NO_WINDOW
        )
        assert proc.stdout is not None
        idx = 0
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if not buf or len(buf) < frame_bytes:
                    break
                k = wanted.get(idx)
                if k is not None:
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3)
                    img = Image.fromarray(np.ascontiguousarray(arr))
                    dest = out_files[k]
                    if dest.suffix.lower() == ".png":
                        img.save(dest, compress_level=6)
                    else:
                        img.save(dest, quality=int(quality), subsampling=1)
                idx += 1
        finally:
            proc.stdout.close()
            proc.wait()
            errf.seek(0)
            err = errf.read().decode("utf-8", "replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"导出关键帧失败：{path}\n{err}")

    missing = len(indices) - sum(1 for p in out_files if p.exists())
    if missing:
        raise RuntimeError(f"有 {missing} 个关键帧没能导出（视频解码出的帧数少于预期）")


def _export_gif(path: Path, indices: Sequence[int], out_size: tuple[int, int],
                out_files: Sequence[Path], quality: int) -> None:
    if not _HAS_PIL:
        raise RuntimeError("处理 GIF 需要 Pillow：pip install Pillow")
    out_w, out_h = out_size
    with Image.open(path) as im:
        for idx, dest in zip(indices, out_files):
            im.seek(idx)
            img = im.convert("RGB")
            if img.size != (out_w, out_h):
                img = img.resize((out_w, out_h), Image.LANCZOS)
            if dest.suffix.lower() == ".png":
                img.save(dest, compress_level=6)
            else:
                img.save(dest, quality=int(quality), subsampling=1)


# --------------------------------------------------------------------- 总览图
# 优先挑支持中文的字体（格子里可能有中文字幕标签），找不到就退回拉丁字体
_FONT_CJK = ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc",
             "Deng.ttf", "NotoSansCJK-Regular.ttc", "PingFang.ttc")
_FONT_LATIN = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Helvetica.ttf")


@lru_cache(maxsize=1)
def _has_cjk_font() -> bool:
    if not _HAS_PIL:
        return False
    for name in _FONT_CJK:
        try:
            ImageFont.truetype(name, 20)
            return True
        except Exception:
            continue
    return False


@lru_cache(maxsize=64)
def _load_font(size: int):
    """按字号取字体，中文优先"""
    size = max(8, int(size))
    for name in (_FONT_CJK if _has_cjk_font() else ()) + _FONT_LATIN:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _grid_cols(n: int, cell_aspect: float = 16 / 9) -> int:
    """
    给 n 格选一个列数，让整张图最接近正方形（同样多的像素，越接近正方形每格越大）。

    注意不能用 ceil(sqrt(n))：横屏帧是宽扁的，同样格数下选列数偏大的网格
    （例如 4 格选 2x2）整图会长得又宽又扁，反而比 3x2 占更多宽度、每格更小。
    这里直接对候选列数算一遍整图宽高比，取最接近 1 的那个。
    """
    if n <= 1:
        return 1
    best_cols, best_cost = 1, None
    for c in range(1, n + 1):
        r = math.ceil(n / c)
        w = c * cell_aspect
        h = float(r)
        cost = max(w / h, h / w)
        if best_cost is None or cost < best_cost - 1e-9:
            best_cols, best_cost = c, cost
    return best_cols


def build_overview_sheets(
    items: Sequence[tuple[Path, str]],
    out_dir: Path,
    stem: str,
    cols: int = 0,
    target_width: int = SHEET_WIDTH,
    max_cells: int = SHEET_MAX_CELLS,
    max_sheets: int = 1,
    per_sheet_cap: int = 40,
    min_cell: int = 280,
    cell_width: int = 0,
    quality: int = 95,
) -> dict:
    """
    把关键帧拼成总览图 —— 帧多的时候拼成**多张**，而不是一张塞满小格。

    图面上只有「关键帧网格 + 每格下方的 #序号/时间戳」，**不印任何元数据条**
    （尺寸、时长、帧率、阈值这些放在 keyframes.json 里，理解画面不需要它们占地方，
    省下的高度都留给格子本身）。

    为什么要分张：读图工具会把每张图整体压到长边约 1568px 再交给模型，
    所以**每格能分到多少像素只取决于「一张图里有多少格」**，跟把图做多大基本无关。
    一张塞 40 格，每格只剩 ~250px 宽、小字全糊；拆成 3 张每张 12 格，
    每格能拿到 ~380px，字幕标题都读得出来。所以清晰度的旋钮是「每张几张」，
    不是「图做多大」。

    分配策略：先按 max_cells 把帧均匀抽稀到 max_sheets * max_cells 以内，
    再把总格数尽量平均地切成若干张（每张的格数差不超过 1），这样每张的
    清晰度一致，不会出现前面清楚后面糊。

    返回 {"sheets": [...], "size", "cols", "rows", "cells", "total", "sheet_count",
          "cell_width", "path"}；失败返回 {}。`sheets` 每项为
    {"path", "size", "cols", "rows", "cells", "first_index", "last_index"}。
    """
    if not _HAS_PIL or not items:
        return {}

    total = len(items)
    max_sheets = max(1, int(max_sheets))
    per_sheet_cap = max(1, int(per_sheet_cap))
    # 一张图最多放多少格：取「用户指定的 max_cells」和「单张安全上限」的较小值
    cap = per_sheet_cap
    if max_cells and max_cells > 0:
        cap = min(cap, int(max_cells))

    # 总容量兜住：帧数超过 张数 x 每张上限 就按时间均匀抽稀
    capacity = max_sheets * cap
    if total > capacity:
        pick = np.linspace(0, total - 1, int(capacity)).round().astype(int)
        items = [items[i] for i in sorted(set(pick.tolist()))]
    n = len(items)

    # 尽量平均地分组（每张最多差 1 格），并按时间顺序切
    sheet_count = min(max_sheets, max(1, math.ceil(n / cap)))
    base = n // sheet_count
    extra = n % sheet_count
    groups: list[list[int]] = []          # 每张图放哪些「全局序号」，序号即时间顺序
    cursor = 0
    for g in range(sheet_count):
        take = base + (1 if g < extra else 0)
        if take <= 0:
            take = 1
            cursor = min(cursor, max(0, n - 1))
        chunk_idx = list(range(cursor, min(cursor + take, n)))
        if chunk_idx:
            groups.append(chunk_idx)
        cursor += take

    # 先看一眼帧图自身有多大 —— 格子不用超过它，超了只是插值放大、白占地方
    with Image.open(items[0][0]) as im0:
        src_w = im0.width

    # 每格宽度按「目标图宽 - 边距」反算，并卡在原始帧宽以内
    def _cell_w_for(cols_: int) -> int:
        pad_ = max(10, target_width // 300)
        gap_ = max(8, target_width // 500)
        if cell_width and cell_width > 0:
            w = int(cell_width)
        else:
            w = max(int(min_cell),
                    (int(target_width) - pad_ * 2 - gap_ * (cols_ - 1)) // cols_)
        return max(1, min(w, max(1, src_w), int(SHEET_MAX_CELL_WIDTH)))

    pad = max(10, target_width // 300)
    gap = max(8, target_width // 500)

    sheets: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    use_suffix = len(groups) > 1

    for gi, chunk_idx in enumerate(groups, start=1):
        chunk = [items[i] for i in chunk_idx]
        k = len(chunk)
        # 列数：优先听用户的；没指定就按「整图接近正方形」自动算，且不追求 >4 列
        if cols and cols > 0:
            c = max(1, min(int(cols), k))
        else:
            c = max(1, min(_grid_cols(k), 4))
        cw = _cell_w_for(c)

        thumbs = []
        for p, label in chunk:
            with Image.open(p) as im:
                im = im.convert("RGB")
                h = max(1, int(round(im.height * cw / im.width)))
                thumbs.append((im.resize((cw, h), Image.LANCZOS), label))
        cell_h = max(t.height for t, _ in thumbs)
        rows = (k + c - 1) // c

        label_size = max(17, min(40, int(cw * 0.052)))
        font_label = _load_font(label_size)
        label_h = int(label_size * 1.9)

        W = pad * 2 + c * cw + gap * (c - 1)
        H = pad * 2 + rows * (cell_h + label_h) + gap * max(0, rows - 1)

        sheet = Image.new("RGB", (W, H), (14, 17, 21))
        draw = ImageDraw.Draw(sheet)

        for i, (thumb, label) in enumerate(thumbs):
            r, cc = divmod(i, c)
            x = pad + cc * (cw + gap)
            yy = pad + r * (cell_h + label_h + gap)
            sheet.paste(thumb, (x, yy))
            draw.text((x + 2, yy + cell_h + int(label_size * 0.3)), label,
                      fill=(206, 214, 222), font=font_label)

        base_name = f"{stem}_overview" if stem else "overview"
        name = f"{base_name}_{gi:02d}.jpg" if use_suffix else f"{base_name}.jpg"
        out_path = out_dir / name
        # 4:4:4 不抽样色度，保证格子里的文字/线条不糊
        sheet.save(out_path, quality=int(quality), subsampling=0, optimize=True)
        sheets.append({
            "path": str(out_path),
            "size": [W, H],
            "cols": c,
            "rows": rows,
            "cells": k,
            "first_index": int(chunk_idx[0]),
            "last_index": int(chunk_idx[-1]),
        })

    first = sheets[0]
    return {
        "path": first["path"],
        "sheets": sheets,
        "sheet_count": len(sheets),
        "size": first["size"],
        "cols": first["cols"],
        "rows": first["rows"],
        "cells": first["cells"],
        "total": total,
        "cell_width": _cell_w_for(first["cols"]),
    }


def build_overview_sheet(
    items: Sequence[tuple[Path, str]],
    out_path: Path,
    cols: int = 0,
    target_width: int = SHEET_WIDTH,
    max_cells: int = SHEET_MAX_CELLS,
    min_cell: int = 280,
    cell_width: int = 0,
    quality: int = 95,
) -> dict:
    """
    【兼容保留】拼一张总览图（单张版本，走 build_overview_sheets 的实现）。

    新代码请用 build_overview_sheets —— 它支持帧多时自动分多张。
    """
    info = build_overview_sheets(
        items, out_path.parent, "",   # stem 空 = 直接用调用方给的文件名
        cols=cols, target_width=target_width,
        max_cells=max_cells, max_sheets=1, per_sheet_cap=max_cells or 40,
        min_cell=min_cell, cell_width=cell_width, quality=quality,
    )
    if not info:
        return {}
    # 单张时沿用调用方指定的文件名，别多出 _01 后缀
    actual = Path(info["path"])
    if actual != out_path and actual.exists():
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(actual, out_path)
            info["path"] = str(out_path)
            info["sheets"][0]["path"] = str(out_path)
        except OSError:
            pass
    return info


# --------------------------------------------------------------------- 主流程
def extract_keyframes(
    src: str | Path,
    output: str | Path | None = None,
    threshold: float = 0.0,
    noise_k: float = DEFAULT_NOISE_K,
    min_time: float = 0.30,
    motion_time: float = 1.5,
    max_frames: int = 200,
    max_width: int = 1280,
    quality: int = 92,
    fmt: str = "jpg",
    analyze_width: int = ANALYZE_WIDTH,
    contact_sheet: bool = True,
    cols: int = 0,
    sheet_width: int = SHEET_WIDTH,
    sheet_max_cells: int = SHEET_MAX_CELLS,
    sheet_max_sheets: int = SHEET_MAX_SHEETS,
    cell_width: int = 0,
    sheet_quality: int = 95,
    sheet_only: bool = False,
    include_last: bool = False,
    verbose: bool = True,
) -> dict:
    """
    抽取单个视频 / GIF 的关键帧。

    返回 dict：原尺寸(width/height)、总时长(duration)、帧率(fps)、
    关键帧清单(keyframes，含序号/时间戳/文件路径)。
    """
    src = Path(src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"文件不存在：{src}")
    if src.suffix.lower() not in _VIDEO_EXT | _GIF_EXT:
        raise ValueError(f"不支持的扩展名：{src.suffix}（支持常见视频格式与 .gif）")

    if output:
        out_dir = Path(output).expanduser().resolve()
    elif sheet_only:
        out_dir = src.parent          # 只出总览图：直接放在视频旁边
    else:
        out_dir = src.with_name(f"{src.stem}_keyframes")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = out_dir / "frames"
    if not sheet_only:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("*"):
            if old.is_file():
                old.unlink()

    if verbose:
        print(f"[1/3] 分析画面变化 …  {src.name}", file=sys.stderr)
    scores, n_frames, info = analyze(src, analyze_width)
    if n_frames <= 0:
        raise RuntimeError(f"没读到任何帧：{src}")

    fps = float(info.get("fps") or 0.0)
    if not fps or fps <= 0:
        fps = 25.0  # 兜底，仅用于时间戳换算

    duration = info.get("duration")
    if not duration:
        duration = n_frames / fps if fps else None

    thr, thr_mode = compute_threshold(scores, threshold, noise_k)
    indices = select_keyframes(
        scores, n_frames, fps, thr,
        min_time=min_time, motion_time=motion_time,
        max_frames=max_frames, include_last=include_last,
    )
    if not indices:
        indices = [0]

    if verbose:
        print(
            f"      尺寸 {info['width']}x{info['height']} · "
            f"{fps:g}fps · {fmt_time(duration)} · {n_frames} 帧\n"
            f"      阈值 {thr:.4f}（{thr_mode}）→ 命中 {len(indices)} 个关键帧",
            file=sys.stderr,
        )
        print("[2/3] 导出关键帧 …", file=sys.stderr)

    # 输出尺寸：不放大，只按 --max-width 缩小
    ow, oh = int(info.get("width") or 0), int(info.get("height") or 0)
    if ow <= 0 or oh <= 0:
        ow, oh = max_width or 1280, 720
    if max_width and max_width > 0 and ow > max_width:
        ratio = max_width / ow
        out_w = _even(int(max_width))
        out_h = _even(max(2, int(round(oh * ratio))))
    else:
        out_w, out_h = _even(ow), _even(oh)

    ext = "png" if fmt == "png" else "jpg"
    keyframes = []
    for n, idx in enumerate(indices, start=1):
        t = idx / fps if fps else 0.0
        name = f"kf_{n:03d}_{t:.3f}s.{ext}"
        dest = frames_dir / name
        keyframes.append({
            "n": n,
            "frame_index": int(idx),
            "time": round(t, 4),
            "file": f"frames/{name}",
            "path": str(dest),
            "diff": round(float(scores[idx - 1]) if idx > 0 and idx - 1 < len(scores) else 0.0, 5),
        })

    # --sheet-only：帧只是拼图的中间产物，落到临时目录，拼完即删
    tmp_ctx = None
    if sheet_only:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="kf_frames_")
        export_dir = Path(tmp_ctx.name)
    else:
        export_dir = frames_dir
    out_files = [export_dir / Path(k["file"]).name for k in keyframes]

    sheet_file = None
    sheet_info = None
    try:
        if src.suffix.lower() in _GIF_EXT:
            _export_gif(src, indices, (out_w, out_h), out_files, quality)
        else:
            _export_video(src, indices, (out_w, out_h), out_files, quality)

        if contact_sheet or sheet_only:
            if verbose:
                print("[3/3] 生成总览图 …", file=sys.stderr)
            items = [(p, f"#{k['n']}  {k['time']:.2f}s")
                     for p, k in zip(out_files, keyframes)]

            sheet_stem = src.stem if sheet_only else ""
            sheet_info = build_overview_sheets(
                items, out_dir, sheet_stem,
                cols=cols, target_width=sheet_width,
                max_cells=sheet_max_cells, max_sheets=sheet_max_sheets,
                per_sheet_cap=SHEET_PER_SHEET_CAP,
                cell_width=cell_width, quality=sheet_quality,
            )
            if sheet_info:
                sheet_file = sheet_info["path"]
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    meta = {
        "source": str(src),
        "name": src.name,
        "type": "gif" if src.suffix.lower() in _GIF_EXT else "video",
        "width": info.get("width"),
        "height": info.get("height"),
        "duration": round(duration, 3) if duration else None,
        "duration_text": fmt_time(duration),
        "fps": round(fps, 3),
        "total_frames": int(n_frames),
        "keyframe_count": len(keyframes),
        "threshold": round(thr, 5),
        "threshold_mode": thr_mode,
        "settings": {
            "min_time": min_time,
            "motion_time": motion_time,
            "max_frames": max_frames,
            "max_width": max_width,
            "format": ext,
            "quality": quality,
        },
        "output_dir": str(out_dir),
        "sheet_only": bool(sheet_only),
        "overview": sheet_file,
        "overviews": [s["path"] for s in (sheet_info or {}).get("sheets", [])],
        "overview_count": len((sheet_info or {}).get("sheets", [])),
        "overview_info": sheet_info,
        "keyframes": keyframes,
    }

    if sheet_only:
        # 帧文件已经删掉了，别留下指向不存在文件的路径
        for kf in keyframes:
            kf.pop("path", None)
            kf.pop("file", None)
    else:
        meta_path = out_dir / "keyframes.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# --------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="抽取视频 / GIF 的关键帧（画面发生实质变化的帧），按时间顺序输出，并返回原始尺寸与总时长。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python extract_keyframes.py demo.mp4\n"
            "  python extract_keyframes.py demo.mp4 -o out --threshold 0.03\n"
            "  python extract_keyframes.py demo.mp4 --max-width 960 --format png\n"
            "  python extract_keyframes.py a.mp4 b.gif --json\n"
        ),
    )
    ap.add_argument("inputs", nargs="+", help="输入文件（视频或 gif），可多个")
    ap.add_argument("-o", "--output", help="输出目录（单个输入时生效；默认 <文件名>_keyframes）")
    ap.add_argument("-t", "--threshold", type=float, default=0.0,
                    help="画面变化的判定阈值 0~1，越大越严格（默认 0 = 自动估计）")
    ap.add_argument("--noise-k", type=float, default=DEFAULT_NOISE_K,
                    help=f"自动阈值的灵敏度倍数（默认 {DEFAULT_NOISE_K}，越小抽得越多）")
    ap.add_argument("--min-time", type=float, default=0.30, help="相邻关键帧最小间隔秒数（默认 0.30）")
    ap.add_argument("--motion-time", type=float, default=1.5,
                    help="画面持续变化时强制抽帧的间隔秒数（默认 1.5）")
    ap.add_argument("--max-frames", type=int, default=200, help="最多输出多少张关键帧（默认 200）")
    ap.add_argument("--max-width", type=int, default=0,
                    help="单帧图片最大宽度，0 = 保持原尺寸（默认 0；用原尺寸拼图最清晰）")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 质量 1-100（默认 92）")
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg", help="输出格式（默认 jpg）")
    ap.add_argument("--analyze-width", type=int, default=ANALYZE_WIDTH,
                    help=f"分析用缩略图宽度（默认 {ANALYZE_WIDTH}）")
    ap.add_argument("--cols", type=int, default=0,
                    help="总览图的列数（默认 0 = 自动取整图最接近正方形的网格；给模型读时固定 3~4）")
    ap.add_argument("--sheet-width", type=int, default=SHEET_WIDTH,
                    help=f"总览图目标宽度，越大每格越清晰（默认 {SHEET_WIDTH}）")
    ap.add_argument("--cell-width", type=int, default=0,
                    help="直接指定总览图里每格的宽度（像素），默认 0 = 按 --sheet-width 反算")
    ap.add_argument("--sheet-quality", type=int, default=95,
                    help="总览图 JPEG 质量（默认 95，始终按 4:4:4 不抽样色度保存）")
    ap.add_argument("--sheet-max-cells", type=int, default=SHEET_MAX_CELLS,
                    help=f"总览图每张最多放多少格，超了分到下一张（默认 {SHEET_MAX_CELLS}，0 = 不限制）")
    ap.add_argument("--sheet-max-sheets", type=int, default=SHEET_MAX_SHEETS,
                    help=f"最多生成几张总览图（默认 {SHEET_MAX_SHEETS}）；关键帧更多时按时间均匀抽稀")
    ap.add_argument("--sheet-only", action="store_true",
                    help="只输出总览图（默认放到视频同目录），不另外保存单帧图片和 json")
    ap.add_argument("--no-contact-sheet", action="store_true", help="不生成总览图（只保存单帧图片）")
    ap.add_argument("--include-last", action="store_true", help="额外保留最后一帧")
    ap.add_argument("--json", action="store_true", help="只输出 JSON（每个文件一行），便于程序调用")
    ap.add_argument("-q", "--quiet", action="store_true", help="不打印进度")

    args = ap.parse_args(argv)

    if len(args.inputs) > 1 and args.output:
        print("提示：多个输入时 --output 会被忽略，各自输出到 <文件名>_keyframes/", file=sys.stderr)

    metas, failed = [], 0
    for src in args.inputs:
        try:
            meta = extract_keyframes(
                src,
                output=args.output if len(args.inputs) == 1 else None,
                threshold=args.threshold,
                noise_k=args.noise_k,
                min_time=args.min_time,
                motion_time=args.motion_time,
                max_frames=args.max_frames,
                max_width=args.max_width,
                quality=args.quality,
                fmt=args.format,
                analyze_width=args.analyze_width,
                contact_sheet=not args.no_contact_sheet,
                cols=args.cols,
                sheet_width=args.sheet_width,
                sheet_max_cells=args.sheet_max_cells,
                sheet_max_sheets=args.sheet_max_sheets,
                cell_width=args.cell_width,
                sheet_quality=args.sheet_quality,
                sheet_only=args.sheet_only,
                include_last=args.include_last,
                verbose=not (args.quiet or args.json),
            )
            metas.append(meta)
        except Exception as e:  # 单个文件失败不影响其他文件
            failed += 1
            print(f"处理失败：{src} -> {e}", file=sys.stderr)

    if args.json:
        for m in metas:
            print(json.dumps(m, ensure_ascii=False))
    elif not args.quiet:
        print()
        for m in metas:
            print(f"✔ {m['name']}")
            print(f"  原始尺寸 : {m['width']} x {m['height']}")
            print(f"  总时长   : {m['duration_text']}  ({m['duration']} 秒, {m['fps']:g}fps, {m['total_frames']} 帧)")
            print(f"  关键帧   : {m['keyframe_count']} 张（阈值 {m['threshold']}，{m['threshold_mode']}）")
            if m.get("overview_info"):
                si = m["overview_info"]
                cnt = si.get("sheet_count", 1)
                print(f"  总览图   : {si['sheets'][0]['path'] if cnt == 1 else str(m['output_dir'])}")
                if cnt == 1:
                    s0 = si["sheets"][0]
                    print(f"             {s0['size'][0]} x {s0['size'][1]} px，"
                          f"{s0['cols']} 列 x {s0['rows']} 行，共 {s0['cells']} 格"
                          f"（每格宽 {si['cell_width']}px）")
                else:
                    print(f"             共 {cnt} 张（每张都保证每格看得清，逐张读）")
                    for s in si["sheets"]:
                        print(f"               {s['path']}  —  {s['cells']} 格, "
                              f"{s['cols']}x{s['rows']}, 每格宽 {si['cell_width']}px")
            if m.get("output_dir") and not m.get("sheet_only"):
                print(f"  输出目录 : {m['output_dir']}")
            print()

    return 1 if (failed and failed == len(args.inputs)) else 0


if __name__ == "__main__":
    sys.exit(main())
