"""ATLAS-120k のクラス表とマスクローダ。

ATLAS のマスクは **palette(P) mode PNG で画素値そのものがクラス index**。
CholecSeg8k の「色 → クラス」変換とは逆向きなので、`mask_to_id_map` ではなく
`load_atlas_id_map` を使う（パレットは表示色でしかなく、意味を持たない）。

クラス定義は ATLAS 側の `atlas120k_tools/classes.py`（47 エントリ、id 0-46）を
**値ごと写した静的テーブル**として持つ。`../ATLAS` への import 依存を作らないため。
出所と検証: docs/pipeline/atlas_scene_graph_expansion_plan.md §2。

`id` がマスクの画素値そのもの。`train_id` は学習用に統合した 30 クラス側の id で
（0 = 背景/除外、Omentum→Fat・Aorta→Artery のように潰れる）、**viewer の凡例と
graph ノード名には情報量の多い元クラス（`id` / `name` / `color`）を使う**。

色の検証: 実データ（gastric_surgery/4FHGGFZsPzw のマスク）に埋め込まれたパレットと
47 件中 45 件が完全一致した。残る 2 件は id 45 Pancreas（パレットは (252, 186, 3)）と
id 46 Duodenum（パレット未設定）で、いずれも `train_id == 0` の末尾エントリ。
表示色だけの差なので id→name の対応には影響しない。ここでは classes.py の値を採る。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from surgical_core.viewer.labels import LabelEntry, LabelTable


@dataclass(frozen=True)
class AtlasClass:
    """ATLAS-120k の 1 クラス。

    Attributes:
        id: マスク PNG の画素値。
        train_id: 統合後 30 クラスの id（0 = 背景/評価除外）。
        name: 表示名。
        category: 解剖区分（abdomen / rarp / thorax / special / background）。
        color: 元パレットの表示色 RGB（各 0-255）。
    """
    id: int
    train_id: int
    name: str
    category: str
    color: tuple[int, int, int]


ATLAS_CLASSES: tuple[AtlasClass, ...] = (
    AtlasClass( 0,  0, 'Background'               , 'background' , (0, 0, 0)),
    AtlasClass( 1,  1, 'Tools/camera'             , 'abdomen'    , (255, 255, 255)),
    AtlasClass( 2,  2, 'Vein (major)'             , 'abdomen'    , (0, 0, 255)),
    AtlasClass( 3,  3, 'Artery (major)'           , 'abdomen'    , (255, 0, 0)),
    AtlasClass( 4,  4, 'Nerve (major)'            , 'abdomen'    , (255, 255, 0)),
    AtlasClass( 5,  5, 'Small intestine'          , 'abdomen'    , (0, 255, 0)),
    AtlasClass( 6,  6, 'Colon/rectum'             , 'abdomen'    , (0, 200, 100)),
    AtlasClass( 7,  7, 'Abdominal wall'           , 'abdomen'    , (200, 150, 100)),
    AtlasClass( 8,  8, 'Diaphragm'                , 'abdomen'    , (250, 150, 100)),
    AtlasClass( 9,  9, 'Omentum'                  , 'abdomen'    , (255, 200, 100)),
    AtlasClass(10,  3, 'Aorta'                    , 'abdomen'    , (180, 0, 0)),
    AtlasClass(11,  2, 'Vena cava'                , 'abdomen'    , (0, 0, 180)),
    AtlasClass(12, 10, 'Liver'                    , 'abdomen'    , (150, 100, 50)),
    AtlasClass(13, 11, 'Cystic duct'              , 'abdomen'    , (0, 255, 255)),
    AtlasClass(14, 12, 'Gallbladder'              , 'abdomen'    , (0, 200, 255)),
    AtlasClass(15,  2, 'Hepatic vein'             , 'abdomen'    , (0, 100, 255)),
    AtlasClass(16, 13, 'Hepatic ligament'         , 'abdomen'    , (255, 150, 50)),
    AtlasClass(17, 14, 'Cystic plate'             , 'abdomen'    , (255, 220, 200)),
    AtlasClass(18, 15, 'Stomach'                  , 'abdomen'    , (200, 100, 200)),
    AtlasClass(19, 11, 'Ductus choledochus'       , 'abdomen'    , (144, 238, 144)),
    AtlasClass(20,  9, 'Mesenterium'              , 'abdomen'    , (247, 255, 0)),
    AtlasClass(21, 11, 'Ductus hepaticus'         , 'abdomen'    , (255, 206, 27)),
    AtlasClass(22, 16, 'Spleen'                   , 'abdomen'    , (200, 0, 200)),
    AtlasClass(23, 17, 'Uterus'                   , 'abdomen'    , (255, 0, 150)),
    AtlasClass(24, 18, 'Ovary'                    , 'abdomen'    , (255, 100, 200)),
    AtlasClass(25, 19, 'Oviduct'                  , 'abdomen'    , (200, 100, 255)),
    AtlasClass(26, 20, 'Prostate'                 , 'rarp'       , (150, 0, 100)),
    AtlasClass(27, 21, 'Urethra'                  , 'rarp'       , (255, 200, 255)),
    AtlasClass(28, 22, 'Ligated plexus'           , 'rarp'       , (150, 100, 75)),
    AtlasClass(29, 23, 'Seminal vesicles'         , 'rarp'       , (200, 0, 150)),
    AtlasClass(30, 24, 'Catheter'                 , 'rarp'       , (100, 100, 100)),
    AtlasClass(31, 25, 'Bladder'                  , 'rarp'       , (255, 150, 255)),
    AtlasClass(32,  0, 'Kidney'                   , 'rarp'       , (100, 200, 255)),
    AtlasClass(33, 26, 'Lung'                     , 'thorax'     , (150, 200, 255)),
    AtlasClass(34, 27, 'Airway (bronchus/trachea)', 'thorax'     , (0, 150, 255)),
    AtlasClass(35, 28, 'Esophagus'                , 'thorax'     , (255, 100, 100)),
    AtlasClass(36, 29, 'Pericardium'              , 'thorax'     , (200, 200, 255)),
    AtlasClass(37,  2, 'V azygos'                 , 'thorax'     , (100, 100, 255)),
    AtlasClass(38, 11, 'Thoracic duct'            , 'thorax'     , (0, 255, 150)),
    AtlasClass(39,  4, 'Nerves'                   , 'thorax'     , (255, 255, 100)),
    AtlasClass(40,  0, 'Ureter'                   , 'special'    , (150, 150, 150)),
    AtlasClass(41, 24, 'Non anatomical structures', 'special'    , (50, 50, 50)),
    AtlasClass(42,  0, 'Excluded frames'          , 'special'    , (0, 0, 0)),
    AtlasClass(43,  0, 'Mesocolon'                , 'abdomen'    , (173, 216, 230)),
    AtlasClass(44,  0, 'Adrenal gland'            , 'abdomen'    , (255, 140, 0)),
    AtlasClass(45,  0, 'Pancreas'                 , 'abdomen'    , (223, 3, 252)),
    AtlasClass(46,  0, 'Duodenum'                 , 'abdomen'    , (0, 80, 100)),
)

ATLAS_CLASS_BY_ID: dict[int, AtlasClass] = {c.id: c for c in ATLAS_CLASSES}

# 背景扱い（描画は暗灰・graph ノード/凡例から除外）:
#   0 = Background、42 = Excluded frames（フレーム全体を評価から外す印で解剖ではない）。
# Tools/camera(1) と Non anatomical structures(41) は前景のまま残す — 実在する物体で、
# シーングラフ上も器具ノードとして意味がある。
ATLAS_BACKGROUND_IDS: frozenset[int] = frozenset({0, 42})


def atlas_gt_table() -> LabelTable:
    """ATLAS-120k GT クラス表（元 47 クラスの意味名＋固定色）。"""
    entries = {
        c.id: LabelEntry(c.name, [v / 255.0 for v in c.color])
        for c in ATLAS_CLASSES
        if c.id not in ATLAS_BACKGROUND_IDS
    }
    return LabelTable(entries=entries, background_ids=set(ATLAS_BACKGROUND_IDS))


def load_atlas_id_map(path: str | Path) -> np.ndarray:
    """ATLAS のマスク PNG を (H, W) の int クラス index 配列として読む。

    palette(P) mode の元マスクと、extract_atlas_frames.py が書き出した L mode の
    index PNG の両方を受ける。どちらも画素値がクラス index なので、パレットは
    参照せず `np.array` の値をそのまま返す。

    Args:
        path: マスク PNG のパス。

    Returns:
        (H, W) の int32 配列。値は ATLAS のクラス id。

    Raises:
        ValueError: RGB など、画素値がクラス index でないモードだったとき。
    """
    img = Image.open(path)
    if img.mode not in ("P", "L"):
        raise ValueError(
            f"{path}: mode={img.mode} のマスクは画素値がクラス index ではない。"
            "ATLAS のマスクは palette(P) か、抽出済みの L mode index PNG のはず。"
        )
    return np.asarray(img, dtype=np.int32)


def unknown_atlas_ids(id_map: np.ndarray) -> list[int]:
    """id_map のうちクラス表に無い id を昇順で返す（空なら全て既知）。

    落とさず通して警告するための検査。呼び出し側は `LabelTable.name` の
    フォールバック表示のままログに出す。
    """
    return sorted(int(v) for v in np.unique(id_map) if int(v) not in ATLAS_CLASS_BY_ID)
