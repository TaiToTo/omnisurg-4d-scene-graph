"""LabelTable: ラベル id → (name, color)。cholecseg8k / instance の 2 プロバイダ。

seg/graph ビルダーはこの表だけに依存し、ラベル源（GT クラス or sam3d インスタンス）を
区別しない。docs/viewer/viewer_export_refactor.md §2/§4。
"""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from surgical_core.cholec import COLOR_TO_CLASS
from .palette import instance_color, BACKGROUND_COLOR


@dataclass(frozen=True)
class LabelEntry:
    """1 ラベルの表示属性（名前と色）。

    Attributes:
        name: 凡例/graph ノードに出す表示名。
        color: float RGB（各 0..1、長さ 3）。
    """
    name: str
    color: list[float]


@dataclass
class LabelTable:
    """id → LabelEntry の対応表 ＋ 背景 id 集合。

    Attributes:
        entries: 前景 id のみの {id: LabelEntry}。
        background_ids: 背景扱いの id（描画は暗灰・graph ノード/凡例から除外）。
    """
    entries: dict[int, LabelEntry]
    background_ids: set[int]

    def name(self, i: int) -> str:
        e = self.entries.get(int(i))
        return e.name if e else f"id {int(i)}"

    def color(self, i: int) -> list[float]:
        e = self.entries.get(int(i))
        return list(e.color) if e else list(BACKGROUND_COLOR)

    def is_background(self, i: int) -> bool:
        return int(i) in self.background_ids

    def present_ids(self, id_map: np.ndarray) -> list[int]:
        """id_map に実在する前景 id を昇順で返す（背景は除外）。"""
        return [int(v) for v in np.unique(id_map) if int(v) not in self.background_ids]


def cholec_gt_table() -> LabelTable:
    """CholecSeg8k クラス表（意味名＋固定色）。背景 id=0。"""
    # COLOR_TO_CLASS: rgb(uint8) -> (class_id, display_name)。id ごとに最初の rgb を採用。
    entries = {}
    for rgb, (cid, name) in COLOR_TO_CLASS.items():
        if cid != 0 and cid not in entries:
            entries[cid] = LabelEntry(name, [c / 255.0 for c in rgb])
    return LabelTable(entries=entries, background_ids={0})


def instance_table(ids: Iterable[int]) -> LabelTable:
    """インスタンス id 列 → 汎用名 `obj N`＋インスタンス色の表。0=背景。"""
    entries = {int(i): LabelEntry(f"obj {int(i)}", instance_color(int(i)))
               for i in ids if int(i) != 0}
    return LabelTable(entries=entries, background_ids={0})


def remap_to_instances(
    label_maps: list[np.ndarray],
) -> tuple[list[np.ndarray], LabelTable, dict[int, int]]:
    """生インスタンスラベル列(-1/0..) をフレーム横断で 1.. へ詰める。

    sam3d の生ラベルは -1=未割当・0..=インスタンス。viewer 規約に合わせ背景=0 を予約し、
    実インスタンスを 1.. へ連番化（同 ID = 同色になるようフレーム横断で一意に割当）。

    Args:
        label_maps: list[(H,W) int]（-1=未割当）。
    Returns:
        (remapped, table, orig_to_new):
          remapped=list[(H,W) int]（0=背景, 1..）、table=LabelTable、
          orig_to_new=dict{生 id -> 新 id}（凡例に元 ID を残したい時用）。
    """
    # 1) 全フレームを走査し、実インスタンスの生 id を一意に集める。
    #    -1（未割当）は捨てる。0 以上だけがインスタンス。
    ids = set()
    for L in label_maps:
        ids.update(int(v) for v in np.unique(L) if v >= 0)

    # 2) 生 id を昇順に並べ、1 から詰めた新 id を割り当てる（0 は背景に予約）。
    #    全フレーム共通の 1 枚の辞書なので、同じ生 id はどのフレームでも同じ新 id
    #    ＝同じ色になる（時系列の同一インスタンスを色で追える）。
    orig_to_new = {orig: i + 1 for i, orig in enumerate(sorted(ids))}

    # 3) 各フレームを新 id へ塗り替える。初期値 0（背景）で埋め、生 id の画素だけ
    #    新 id で上書き。-1 や未出現 id はそのまま 0（背景）に落ちる。
    remapped = []
    for L in label_maps:
        r = np.zeros_like(L, dtype=np.int64)
        for orig, new in orig_to_new.items():
            r[L == orig] = new
        remapped.append(r)

    # table は新 id（1..）に対する `obj N`＋色。orig_to_new は凡例で元 id を出す時用。
    return remapped, instance_table(orig_to_new.values()), orig_to_new
