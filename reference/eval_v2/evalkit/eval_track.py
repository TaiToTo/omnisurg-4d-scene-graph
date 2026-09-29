"""sam3d track ラベルを GT で定量評価する A/B 比較ハーネス（cholecseg8k / ATLAS-120k / LapEx）。

`track_sam3.py` が出す `out/<clip>/track_<base>/label_<absframe>.npy`（時系列一貫 ID の
領域ラベル (H,W) int）を、`<root>/<clip>/seg_masks/` の GT クラスマスクと突き合わせて
tracking_improvement_spec §3 の指標を測る。GT は 2 形式に対応:
cholec_gt は `<absframe>_color_mask.png`（色→クラス）、
atlas120k / LapEx は `<absframe>_class.png`（画素値=クラス index）。

**F1-v2（2026-09-07）。v1 のキーは値を 1 つも変えずに、次を足した**（監査
[docs/paper/ipcai2027_eval_audit.md](../docs/paper/ipcai2027_eval_audit.md)）:

  - inst-F1 を 0.50〜0.95（0.05 刻み）の全閾値で出し、その平均 `f1_avg` を持つ
    （F1@0.5 は境界が数十画素ずれても満点のまま ＝ 輪郭の質を見ていない）。
  - SQ（マッチした対の平均 IoU ＝ PQ / F1@0.5）を独立に持つ。マスクの質だけを見る。
  - マッチした対ごとの boundary-F（`inst_bf`。インスタンス単位の輪郭の質）。
  - boundary-F / raw boundary P・R を許容 1・2・3・5 画素で出す（2 は v1 と同じ）。
  - 土俵 4 種を 1 回で出す: `full`（v1 そのまま）/ `labeled`（GT 背景を除く）/
    `tissue`（GT 器具クラスを除く）/ `labeled_tissue`。これまで別スクリプト
    （`tissue_only_metrics.py` / `metric_fairness.py`）が凍結の外で作っていた土俵を
    凍結の中へ入れた（＝ 別の入口を作らない）。
  - TP / |GT| / |pred| の生カウント（pooled F1 を後から組めるように）。
  - VI が NaN になるフレーム（GT が無視クラスだけ）を集計から除く（v1 は NaN が伝播）。

指標:
  - GT-mIoU       : 各 track 領域を最大重なり GT クラスへ割当→クラス map を作り per-class IoU。
                    **過分割を罰しない**（割っても多数決で同じクラスに戻る）。
  - GT-mDice      : 同じ割当てから per-class Dice = 2TP/(2TP+FP+FN)。
  - boundary-F    : 上記クラス map の境界と GT クラス境界の F 値（tol 画素の許容。
                    境界は両側 1 画素ずつの 2 画素幅で描くので実効許容は片側 tol+1）。
  - boundary-R-raw: 割当て**前**の生領域境界が GT クラス境界を回収できている割合
                    （assignment-free boundary recall）。過分割による余分な境界は罰しない。
                    参考として precision (boundary-P-raw) も併記。
  - underseg-error: superpixel 標準の corrected undersegmentation error。
                    UE = Σ_S Σ_{R∩S≠∅} min(|R∩S|, |R∖S|) / Σ_S |S|
  - overseg       : GT 1 クラスに重なる track 片の数（過分割。少ないほど良い）。
  - time-IoU      : 同一 obj_id の隣接フレーム画素 IoU（時系列一貫性。高いほど良い）。
  - inst-F1@t / PQ / SQ: GT の連結成分を「表面インスタンス」とみなし、予測領域と IoU で
                    一意にマッチして測る（過分割=precision 低、過小分割=recall 低）。

評価ドメインは depth 有効画素（depth>0 & finite）に限定（パイプラインが扱う領域）。
クラス 0=background は GT-mIoU/overseg の集計から除外（組織・器具に注目）。

Usage:
    python eval_track.py --root ../outputs/cholec_gt --clip VID01_s15_80_crop \
        --track_dir out/VID01_s15_80_crop/track_rgb --tag baseline --dataset cholec
    python eval_track.py --root ../outputs/atlas --clip lar___7H7G-4sevQ__gt_0001 \
        --track_dir out_atlas/lar___7H7G-4sevQ__gt_0001/track_rgb --tag atlas_baseline \
        --dataset atlas
"""
import os
import glob
import json
import argparse

import numpy as np
import cv2

from surgical_core.cholec import mask_to_id_map
from surgical_core.atlas import load_atlas_id_map

# 並列に多数流すとき OpenCV のスレッドが競合する（既定は論理コア数）。`CV2_THREADS` で絞る。
if os.environ.get("CV2_THREADS"):
    cv2.setNumThreads(int(os.environ["CV2_THREADS"]))


EVAL_VERSION = 2                  # 指標の版。v1 のキーは v2 でも同じ値を返す（回帰テストで保証）

BACKGROUND = 0   # cholecseg8k の背景クラス id（GT-mIoU/overseg から除外）

# instance-level (panoptic) 指標の設定。EndoLINA の主指標: GT の連結成分を
# 「表面インスタンス」とみなし、予測領域と IoU マッチングして F1 / PQ を測る
# （過分割=precision 低、過小分割=recall 低の両方を罰する。normal が勝つ唯一の土俵）。
INSTANCE_MIN_CC = 300              # 連結成分/予測領域の最小画素数（細片ノイズ除外）
INSTANCE_THRS = (0.5, 0.75)       # v1 から出している閾値（キー名 `f1@0.5` / `f1@0.75` を維持）
INSTANCE_THRS_ALL = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))   # 0.50 … 0.95
BOUNDARY_TOLS = (1, 2, 3, 5)      # boundary-F の許容画素。2 が v1 の値
BOUNDARY_TOL = 2                  # v1 互換のキー（`bf` / `br_raw`）が使う許容
EXTRA_IGNORE = set()              # 追加で無視する GT クラス id（例 {6,7}=connective/blood）。
                                  # instance/VI の GT ドメインから除外。CLI --extra_ignore で設定。

# GT の器具クラス id（`tissue` 土俵で評価ドメインから外す）。**定義の正はここ** —
# `track_sam3.py`（手法側）と `ipcai2027_experiment/scripts/_common.py`（分析側）は
# これと一致することを確かめてから使う。データセットを増やすときはここへ足す。
INSTRUMENT_CLASSES = {
    "cholec": (5, 9),             # cholecseg8k: 5=grasper, 9=l_hook
    "atlas": (1,),                # atlas120k: 1 = Tools/camera
    "lapex": (236, 145, 182),     # LapEx: biforceps / liver_retractor / flat_grasper
}
TISSUE_IGNORE: tuple[int, ...] = ()   # 実行時に `--dataset` / `--instrument_classes` で決まる

# 土俵の名前。`full` は v1 の指標そのもの。
DOMAINS = ("full", "labeled", "tissue", "labeled_tissue")


def _load_depth(root, clip):
    """results.npz の depth (N,H,W) を返す（有効画素ドメイン用）。"""
    z = np.load(os.path.join(root, clip, "exports", "mini_npz", "results.npz"))
    return z["depth"].astype(np.float32)


def _gt_idmap(root, clip, abs_frame, H, W):
    """GT マスク → (H,W) クラス id map。そのフレームに GT が無ければ None。

    データセットで GT の持ち方が違うので両方受ける:

    | dataset   | ファイル名               | 画素値       |
    |-----------|--------------------------|--------------|
    | cholec_gt | `NNNNNN_color_mask.png`  | 色 → クラス  |
    | atlas120k | `NNNNNN_class.png`       | クラス index |

    どちらも背景は 0 なので `BACKGROUND` の扱いは共通。

    Args:
        root: クリップのルート。clip: クリップ名。abs_frame: 絶対フレーム番号。
        H, W: ラベル解像度（GT はここへ nearest で合わせられる）。

    Returns:
        (H,W) int のクラス id map。GT が無ければ None。
    """
    seg_dir = os.path.join(root, clip, "seg_masks")
    color_p = os.path.join(seg_dir, f"{abs_frame:06d}_color_mask.png")
    if os.path.exists(color_p):                      # cholec_gt: 色 → クラス
        idm = mask_to_id_map(color_p, H, W)
        if idm is None:                              # 読めない PNG を黙って「GT 無し」にしない
            raise RuntimeError(f"GT が読めない（壊れた PNG か権限）: {color_p}")
        return idm
    class_p = os.path.join(seg_dir, f"{abs_frame:06d}_class.png")
    if os.path.exists(class_p):                      # atlas120k: 画素値 = クラス index
        idm = load_atlas_id_map(class_p)
        if idm.shape[:2] != (H, W):                  # ラベルは深度解像度なので合わせる
            idm = cv2.resize(idm.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
        return idm.astype(np.int32)
    return None


def _boundary(idmap):
    """(H,W) int ラベル→境界 bool（4近傍で隣と値が違う画素）。"""
    b = np.zeros(idmap.shape, dtype=bool)
    b[:, :-1] |= idmap[:, :-1] != idmap[:, 1:]
    b[:, 1:] |= idmap[:, :-1] != idmap[:, 1:]
    b[:-1, :] |= idmap[:-1, :] != idmap[1:, :]
    b[1:, :] |= idmap[:-1, :] != idmap[1:, :]
    return b


def _dilate(b, tol):
    """bool 境界を (2tol+1)² の箱で膨らませる（許容 tol 画素）。"""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    return cv2.dilate(b.astype(np.uint8), k).astype(bool)


def _boundary_prf(pb, gb, tol):
    """境界 bool 対の precision / recall / F（tol 画素許容）。どちらかが空なら (0,0,0)。"""
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0, 0.0, 0.0
    pb_d = _dilate(pb, tol)
    gb_d = _dilate(gb, tol)
    prec = (pb & gb_d).sum() / pb.sum()      # pred 境界のうち GT 境界近傍にある割合
    rec = (gb & pb_d).sum() / gb.sum()       # GT 境界のうち pred 境界近傍にある割合
    f = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f)


def _boundary_f(pred_cls, gt_cls, valid, tol=2):
    """pred クラス map と GT クラス境界の boundary precision/recall/F（tol 画素許容）。"""
    pb = _boundary(pred_cls) & valid
    gb = _boundary(gt_cls) & valid
    return _boundary_prf(pb, gb, tol)


def _boundary_recall_raw(lab, gt_cls, valid, tol=2):
    """生 track 領域境界（クラス割当て前）で測る GT 境界の recall / precision。

    過分割で境界が増えても recall は下がらない（assignment-free）。precision は
    「生領域境界のうち GT 境界近傍にある割合」で、過分割の余分な境界量の参考値。

    Args:
        lab: (H,W) int 生 track ラベル（-1=未割当。未割当との境も境界に数える）。
        gt_cls: (H,W) GT クラス id map。
        valid: (H,W) bool 評価ドメイン。
        tol: 一致許容画素数。
    Returns:
        (precision, recall) float。どちらかの境界が空なら (0,0)。
    """
    pb = _boundary(lab) & valid
    gb = _boundary(gt_cls) & valid
    prec, rec, _ = _boundary_prf(pb, gb, tol)
    return prec, rec


def _underseg_error(lab, gt_cls, valid):
    """corrected undersegmentation error（superpixel 評価の標準形）。

    UE = Σ_S Σ_{R: R∩S≠∅} min(|R∩S|, |R∖S|) / Σ_S |S|。
    S は GT 非背景クラス領域、R は生 track 領域（-1 未割当は R に含めない＝
    どこにも merge されていない画素は漏れとして数えない）。全て有効画素内で数える。

    Returns:
        UE float（0=境界を跨ぐ merge なし。小さいほど良い）。GT 非背景が空なら 0。
    """
    total_gt = 0
    leak = 0
    for c in np.unique(gt_cls[valid]):
        if c == BACKGROUND:
            continue
        smask = (gt_cls == c) & valid
        sarea = int(smask.sum())
        if sarea == 0:
            continue
        total_gt += sarea
        for r in np.unique(lab[smask]):
            if r < 0:
                continue
            rmask = (lab == r) & valid
            inter = int((rmask & smask).sum())
            out = int(rmask.sum()) - inter
            leak += min(inter, out)
    return float(leak / total_gt) if total_gt else 0.0


def _region_to_class(lab, gt, valid):
    """各 track 領域(obj_id)を、有効画素での最大重なり GT クラスへ割当。

    Returns:
        dict{obj_id -> class_id}。背景しか重ならない領域も割当（後段で除外可）。
    """
    assign = {}
    for r in np.unique(lab):
        if r < 0:
            continue
        m = (lab == r) & valid
        if not m.any():
            continue
        cls = gt[m]
        assign[int(r)] = int(np.bincount(cls, minlength=13).argmax())
    return assign


def _gt_instances(gt, valid):
    """GT を「表面インスタンス」に分解: 各非無視クラスの連結成分（>=MIN_CC）。

    Args:
        gt: (H,W) GT クラス id map。
        valid: (H,W) bool 評価ドメイン。
    Returns:
        list[(H,W) bool] 各インスタンスのマスク。
    """
    ign = {BACKGROUND} | EXTRA_IGNORE
    out = []
    for c in np.unique(gt[valid]):
        if int(c) in ign:
            continue
        n, lbl = cv2.connectedComponents(((gt == c) & valid).astype(np.uint8))
        for i in range(1, n):
            m = lbl == i
            if int(m.sum()) >= INSTANCE_MIN_CC:
                out.append(m)
    return out


def _pred_regions(lab, valid):
    """予測ラベルの各領域（>=MIN_CC）を bool マスクの list で返す。"""
    return [(lab == r) & valid
            for r in np.unique(lab[valid & (lab >= 0)])
            if int(((lab == r) & valid).sum()) >= INSTANCE_MIN_CC]


def _match_ious(gts, preds):
    """GT インスタンス ↔ 予測領域を IoU 降順で一意にグリーディ・マッチ。

    Returns:
        (matched_ious, n_gt, n_pred)。matched_ious は各マッチの IoU（降順）。
    """
    pairs = []
    for gi, g in enumerate(gts):
        for pj, p in enumerate(preds):
            u = int((g | p).sum())
            if u:
                iou = int((g & p).sum()) / u
                if iou > 0:
                    pairs.append((iou, gi, pj))
    pairs.sort(reverse=True)
    gused, pused, ious = set(), set(), []
    for iou, gi, pj in pairs:
        if gi in gused or pj in pused:
            continue
        gused.add(gi); pused.add(pj); ious.append(iou)
    return ious, len(gts), len(preds)


# ── v2: 分割表による同じマッチング（速い版。v1 と同じ結果を返すことをテストで保証） ──

def _gt_instance_map(gt, valid):
    """`_gt_instances` と同じ順序・同じ足切りで、インスタンス id map (H,W) と面積列を返す。

    id は 1 始まり（0 = どのインスタンスでもない）。順序は `_gt_instances` の list と同じ
    （クラス昇順 → 連結成分ラベル昇順）なので、添字 gi は両実装で一致する。
    """
    ign = {BACKGROUND} | EXTRA_IGNORE
    gid = np.zeros(gt.shape, np.int32)
    areas = []
    for c in np.unique(gt[valid]):
        if int(c) in ign:
            continue
        n, lbl = cv2.connectedComponents(((gt == c) & valid).astype(np.uint8))
        if n <= 1:
            continue
        cnt = np.bincount(lbl.ravel(), minlength=n)
        for i in range(1, n):
            if int(cnt[i]) >= INSTANCE_MIN_CC:
                areas.append(int(cnt[i]))
                gid[lbl == i] = len(areas)
    return gid, areas


def _pred_region_map(lab, valid):
    """`_pred_regions` と同じ順序・同じ足切りで、領域 id map (H,W)・面積列・元 id 列を返す。"""
    L = np.where(valid & (lab >= 0), lab, -1)
    ids, cnts = np.unique(L[L >= 0], return_counts=True)
    keep = cnts >= INSTANCE_MIN_CC
    ids, cnts = ids[keep], cnts[keep]
    pid = np.zeros(lab.shape, np.int32)
    if len(ids):
        lut = np.zeros(int(L.max()) + 2, np.int32)     # 元 id → 1 始まりの添字
        lut[ids] = np.arange(1, len(ids) + 1)
        pid = np.where(L >= 0, lut[np.maximum(L, 0)], 0).astype(np.int32)
    return pid, [int(c) for c in cnts], [int(i) for i in ids]


def _match_pairs(gid, g_areas, pid, p_areas):
    """分割表から IoU を引き、v1 `_match_ious` と同じ規則（IoU 降順・同点は添字降順）で
    一意にグリーディ・マッチする。

    Returns:
        list[(iou, gi, pj)]（マッチ順）。
    """
    G, P = len(g_areas), len(p_areas)
    if G == 0 or P == 0:
        return []
    joint = np.bincount((gid.astype(np.int64) * (P + 1) + pid).ravel(),
                        minlength=(G + 1) * (P + 1)).reshape(G + 1, P + 1)
    inter = joint[1:, 1:]
    pairs = []
    for gi in range(G):
        for pj in range(P):
            i = int(inter[gi, pj])
            if i > 0:
                u = g_areas[gi] + p_areas[pj] - i
                pairs.append((i / u, gi, pj))
    pairs.sort(reverse=True)
    gused, pused, out = set(), set(), []
    for iou, gi, pj in pairs:
        if gi in gused or pj in pused:
            continue
        gused.add(gi); pused.add(pj); out.append((iou, gi, pj))
    return out


def _instance_boundary_f(gid, pid, matched, valid, tol=BOUNDARY_TOL):
    """マッチした対ごとの boundary-F（インスタンス単位の輪郭の質）の平均。対が無ければ None。"""
    vals = []
    for _iou, gi, pj in matched:
        g = (gid == gi + 1)
        p = (pid == pj + 1)
        _, _, f = _boundary_prf(_boundary(p) & valid, _boundary(g) & valid, tol)
        vals.append(f)
    return float(np.mean(vals)) if vals else None


def instance_metrics(lab, gt, valid):
    """1 フレームの instance-F1（各 IoU 閾値）・PQ・SQ・生カウントを返す（GT インスタンス無しは None）。

    F1@t: マッチ IoU>=t を TP、|GT|-TP を FN、|pred|-TP を FP とした F1。
    PQ  : Σ_{IoU>=0.5} IoU / (TP + 0.5 FP + 0.5 FN)（panoptic quality）。
    SQ  : Σ_{IoU>=0.5} IoU / TP（マッチした対の平均 IoU。TP=0 なら None）。
    f1_avg: F1@t を t=0.50〜0.95（0.05 刻み）で平均したもの。
    inst_bf: IoU>=0.5 でマッチした対ごとの boundary-F（tol 2）の平均。TP=0 なら None。

    v1 のキー `f1@0.5` / `f1@0.75` / `pq` は v1 と同じ値。
    """
    gid, g_areas = _gt_instance_map(gt, valid)
    if not g_areas:
        return None
    pid, p_areas, _ids = _pred_region_map(lab, valid)
    matched = _match_pairs(gid, g_areas, pid, p_areas)
    ious = [m[0] for m in matched]
    ng, npd = len(g_areas), len(p_areas)
    out = {}
    curve = {}
    for t in INSTANCE_THRS_ALL:
        tp = sum(1 for i in ious if i >= t)
        denom = 2 * tp + (npd - tp) + (ng - tp)
        curve[t] = float(2 * tp / denom) if denom else 0.0
    for t in INSTANCE_THRS:                      # v1 互換キー（同じ式・同じ値）
        out[f"f1@{t}"] = curve[t]
    tps = [i for i in ious if i >= 0.5]
    tp = len(tps)
    denom = tp + 0.5 * (npd - tp) + 0.5 * (ng - tp)
    out["pq"] = float(sum(tps) / denom) if denom else 0.0
    out["sq"] = float(sum(tps) / tp) if tp else None
    out["f1_avg"] = float(np.mean(list(curve.values())))
    out["f1_curve"] = {f"{t:.2f}": v for t, v in curve.items()}
    out["tp"] = tp
    out["n_gt"] = ng
    out["n_pred"] = npd
    out["inst_bf"] = _instance_boundary_f(
        gid, pid, [m for m in matched if m[0] >= 0.5], valid)
    return out


def variation_of_information(lab, gt, valid):
    """予測ラベル X と GT クラス map Y の VI 分解を bits で返す（無視クラス除外）。

    Returns:
        (split, merge) = (H(X|Y), H(Y|X))。split=過分割誤差、merge=過小分割誤差。
    """
    m = valid & ~np.isin(gt, list({BACKGROUND} | EXTRA_IGNORE))
    x = lab[m].astype(np.int64)
    y = gt[m].astype(np.int64)
    if x.size == 0:
        return (float("nan"), float("nan"))
    _, x = np.unique(x, return_inverse=True)
    _, y = np.unique(y, return_inverse=True)
    j = np.zeros((int(x.max()) + 1, int(y.max()) + 1))
    np.add.at(j, (x, y), 1.0)
    j /= x.size
    H = lambda p: float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    hxy = H(j.ravel())
    return (hxy - H(j.sum(0)), hxy - H(j.sum(1)))


def domain_masks(gt, valid):
    """4 つの土俵の評価ドメインを返す。`tissue` 系は `TISSUE_IGNORE` が空なら None。

    | 土俵 | ドメイン | 何が変わるか |
    |---|---|---|
    | `full` | depth 有効画素 | v1 の指標そのもの |
    | `labeled` | ＋ GT が背景でない | 未注釈の上の予測領域が FP にならない（#7a） |
    | `tissue` | ＋ GT が器具でない | 器具インスタンスを数えない（#2） |
    | `labeled_tissue` | 両方 | 論文の主張（組織側）と揃う土俵 |
    """
    named = valid & (gt != BACKGROUND)
    out = {"full": valid, "labeled": named}
    if TISSUE_IGNORE:
        tis = ~np.isin(gt, list(TISSUE_IGNORE))
        out["tissue"] = valid & tis
        out["labeled_tissue"] = named & tis
    else:
        out["tissue"] = None
        out["labeled_tissue"] = None
    return out


def eval_frame(lab, gt, valid):
    """1 フレームの per-class IoU・boundary-F・overseg・instance 指標（4 土俵）を返す。

    Args:
        lab: (H,W) track ラベル（-1=未割当）。
        gt: (H,W) GT クラス id map。
        valid: (H,W) bool 評価ドメイン。
    Returns:
        dict（ious={cls:iou}, bf=(p,r,f), overseg={cls:count}, n_regions=int, inst=...,
              inst_dom={土俵: inst}, bf_tol={tol:(p,r,f)}, br_raw_tol={tol:(p,r)}）。
    """
    assign = _region_to_class(lab, gt, valid)
    # track 領域→割当クラスで pred クラス map を作る（未割当/領域外は背景扱い）
    pred = np.zeros_like(gt)
    for r, c in assign.items():
        pred[(lab == r)] = c

    # per-class IoU / Dice（GT or pred に現れる非背景クラス、同一割当てから両方）
    classes = set(np.unique(gt[valid]).tolist()) | set(np.unique(pred[valid]).tolist())
    classes.discard(BACKGROUND)
    ious = {}
    dices = {}
    for c in classes:
        p = (pred == c) & valid
        g = (gt == c) & valid
        inter = int((p & g).sum())
        union = int((p | g).sum())
        denom = int(p.sum()) + int(g.sum())
        ious[int(c)] = float(inter / union) if union else 0.0
        dices[int(c)] = float(2 * inter / denom) if denom else 0.0

    # overseg: GT 各非背景クラスに「実質重なる」track 片の数
    overseg = {}
    for c in np.unique(gt[valid]):
        if c == BACKGROUND:
            continue
        gmask = (gt == c) & valid
        garea = int(gmask.sum())
        if garea == 0:
            continue
        n = 0
        for r in np.unique(lab[gmask]):
            if r < 0:
                continue
            inter = int(((lab == r) & gmask).sum())
            # GT クラスの 5% 以上 or 200px 以上重なる片だけ数える（細片ノイズ除外）
            if inter >= max(200, 0.05 * garea):
                n += 1
        overseg[int(c)] = n

    # 境界: クラス map（割当後）と生領域（割当前）の両方、許容ごとに
    pb_cls = _boundary(pred) & valid
    pb_raw = _boundary(lab) & valid
    gb = _boundary(gt) & valid
    bf_tol = {tol: _boundary_prf(pb_cls, gb, tol) for tol in BOUNDARY_TOLS}
    br_raw_tol = {tol: _boundary_prf(pb_raw, gb, tol)[:2] for tol in BOUNDARY_TOLS}
    bf = bf_tol[BOUNDARY_TOL]
    br_raw = br_raw_tol[BOUNDARY_TOL]
    ue = _underseg_error(lab, gt, valid)
    n_regions = int((np.unique(lab) >= 0).sum())

    doms = domain_masks(gt, valid)
    inst_dom = {d: (instance_metrics(lab, gt, m) if m is not None else None)
                for d, m in doms.items()}
    inst = inst_dom["full"]
    vi_split, vi_merge = variation_of_information(lab, gt, valid)
    return dict(ious=ious, dices=dices, bf=bf, br_raw=br_raw, underseg=ue,
                overseg=overseg, n_regions=n_regions,
                inst=inst, inst_dom=inst_dom, bf_tol=bf_tol, br_raw_tol=br_raw_tol,
                vi_split=vi_split, vi_merge=vi_merge)


def time_iou(labels, valids):
    """隣接フレームの同一 obj_id 画素 IoU の平均（時系列一貫性）。

    Args:
        labels: list[(H,W) int]（フレーム順）。
        valids: list[(H,W) bool]。
    Returns:
        全 (obj_id, 隣接ペア) 平均 IoU。
    """
    vals = []
    for t in range(len(labels) - 1):
        a, b = labels[t], labels[t + 1]
        v = valids[t] & valids[t + 1]
        ids = set(np.unique(a).tolist()) & set(np.unique(b).tolist())
        for r in ids:
            if r < 0:
                continue
            ma = (a == r) & v
            mb = (b == r) & v
            union = (ma | mb).sum()
            if union:
                vals.append((ma & mb).sum() / union)
    return float(np.mean(vals)) if vals else 0.0


def _inst_out(m):
    """per_frame へ書く instance dict（丸め。None はそのまま）。"""
    if m is None:
        return None
    o = {k: (round(v, 4) if isinstance(v, float) else v)
         for k, v in m.items() if k != "f1_curve"}
    o["f1_curve"] = {k: round(v, 4) for k, v in m["f1_curve"].items()}
    return o


def _r(x, nd=4):
    return None if x is None else round(x, nd)


def _mean_or_none(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def summarize(per_frame, labels, valids, abs_frames_with_gt):
    """per-frame の結果をクリップ 1 本の summary（v1 キー ＋ v2 キー）へ集計する。

    v1 のキーは v1 と同じ集計（各フレーム present クラスの平均をフレーム平均、inst は
    GT インスタンスありフレームの平均）。v2 のキーは同じ流儀で足す。
    """
    def frame_mean(fr, key):
        vs = list(fr[key].values())
        return float(np.mean(vs)) if vs else 0.0

    def mean_of(fn):
        return float(np.mean([fn(fr) for fr in per_frame])) if per_frame else 0.0

    miou = mean_of(lambda fr: frame_mean(fr, "ious"))
    mdice = mean_of(lambda fr: frame_mean(fr, "dices"))
    bf = mean_of(lambda fr: fr["bf"][2])
    bprec = mean_of(lambda fr: fr["bf"][0])
    brec = mean_of(lambda fr: fr["bf"][1])
    bprec_raw = mean_of(lambda fr: fr["br_raw"][0])
    brec_raw = mean_of(lambda fr: fr["br_raw"][1])
    underseg = mean_of(lambda fr: fr["underseg"])
    overseg_vals = [v for fr in per_frame for v in fr["overseg"].values()]
    overseg = float(np.mean(overseg_vals)) if overseg_vals else 0.0
    overseg_max = int(np.max(overseg_vals)) if overseg_vals else 0
    n_regions = float(np.mean([fr["n_regions"] for fr in per_frame])) if per_frame else 0.0
    tiou = time_iou(labels, valids)

    # instance-level (panoptic) の集計。inst は GT インスタンスありフレームのみ。
    inst_frames = [fr["inst"] for fr in per_frame if fr["inst"] is not None]

    def inst_mean(key):
        return float(np.mean([f[key] for f in inst_frames])) if inst_frames else 0.0

    inst_f1_50 = inst_mean("f1@0.5")
    inst_f1_75 = inst_mean("f1@0.75")
    pq = inst_mean("pq")
    # VI: GT が無視クラスだけのフレームは NaN になる。v1 は NaN が伝播したので、
    # 有限なフレームだけで平均し、使ったフレーム数を残す。
    vi_s, n_vi = _mean_or_none([fr["vi_split"] for fr in per_frame])
    vi_m, _ = _mean_or_none([fr["vi_merge"] for fr in per_frame])

    summary = dict(
        n_frames=len(labels), n_gt_frames=len(per_frame),
        GT_mIoU=round(miou, 4), GT_mDice=round(mdice, 4),
        boundary_F=round(bf, 4),
        boundary_P=round(bprec, 4), boundary_R=round(brec, 4),
        boundary_P_raw=round(bprec_raw, 4), boundary_R_raw=round(brec_raw, 4),
        underseg_error=round(underseg, 4),
        overseg_mean=round(overseg, 3), overseg_max=overseg_max,
        n_regions_mean=round(n_regions, 2), time_IoU=round(tiou, 4),
        inst_F1_50=round(inst_f1_50, 4), inst_F1_75=round(inst_f1_75, 4),
        PQ=round(pq, 4),
        VI_split=_r(vi_s), VI_merge=_r(vi_m), n_vi_frames=n_vi,
        n_inst_frames=len(inst_frames),
        extra_ignore=sorted(EXTRA_IGNORE),
    )

    # ── v2 ──
    summary["eval_version"] = EVAL_VERSION
    summary["tissue_ignore"] = list(TISSUE_IGNORE)
    # 境界を許容ごとに（クラス map と生領域）
    for tol in BOUNDARY_TOLS:
        summary[f"boundary_F_tol{tol}"] = round(mean_of(lambda fr, t=tol: fr["bf_tol"][t][2]), 4)
        summary[f"boundary_P_raw_tol{tol}"] = round(
            mean_of(lambda fr, t=tol: fr["br_raw_tol"][t][0]), 4)
        summary[f"boundary_R_raw_tol{tol}"] = round(
            mean_of(lambda fr, t=tol: fr["br_raw_tol"][t][1]), 4)
    # 土俵ごとの instance 指標
    for d in DOMAINS:
        frs = [fr["inst_dom"][d] for fr in per_frame
               if fr["inst_dom"].get(d) is not None]
        suf = "" if d == "full" else f"_{d}"
        if not frs:
            # **`full` は v1 の振る舞い（0.0 で集計して先へ進む）を保つ。** v1 の
            # `inst_mean` は空 list に 0.0 を返していたので、GT インスタンスが 1 つも
            # 立たないクリップも 0.0 として平均に入った。ここを None にすると
            # `eval_gt_clips.mean()`（`inst_F1_avg` は allow_none でない）が止まり、
            # **その 1 本のせいで他の全クリップの集計ごと落ちる**。
            # 土俵つきは「その土俵が無い」ことを表す None のまま（allow_none のキー）。
            zero = 0.0 if d == "full" else None
            for k in ("inst_F1_50", "inst_F1_75", "inst_F1_avg", "PQ"):
                summary.setdefault(f"{k}{suf}", zero)
            for k in ("SQ", "inst_BF"):     # マッチが 1 つも無いので定義できない
                summary.setdefault(f"{k}{suf}", None)
            # **数のキーもここで必ず書く。** 書き忘れると集計側（`SUMMARY_KEYS` は
            # `n_gt_inst_mean` を allow_none でないキーとして引く）が素の KeyError で死ぬ。
            summary[f"n_gt_inst_mean{suf}"] = zero
            summary[f"n_pred_inst_mean{suf}"] = zero
            summary[f"n_SQ_frames{suf}"] = 0
            summary[f"n_BF_frames{suf}"] = 0
            summary[f"n_inst_frames{suf}"] = 0
            summary[f"inst_F1_curve{suf}"] = None
            summary[f"inst_pooled{suf}"] = None
            continue
        f1_50 = float(np.mean([f["f1@0.5"] for f in frs]))
        f1_75 = float(np.mean([f["f1@0.75"] for f in frs]))
        f1_avg = float(np.mean([f["f1_avg"] for f in frs]))
        pq_d = float(np.mean([f["pq"] for f in frs]))
        sq, n_sq = _mean_or_none([f["sq"] for f in frs])
        ibf, n_ibf = _mean_or_none([f["inst_bf"] for f in frs])
        if d != "full":
            summary[f"inst_F1_50{suf}"] = round(f1_50, 4)
            summary[f"inst_F1_75{suf}"] = round(f1_75, 4)
            summary[f"PQ{suf}"] = round(pq_d, 4)
        summary[f"inst_F1_avg{suf}"] = round(f1_avg, 4)
        summary[f"SQ{suf}"] = _r(sq)
        summary[f"n_SQ_frames{suf}"] = n_sq
        summary[f"inst_BF{suf}"] = _r(ibf)
        summary[f"n_BF_frames{suf}"] = n_ibf
        summary[f"n_inst_frames{suf}"] = len(frs)
        summary[f"inst_F1_curve{suf}"] = {
            k: round(float(np.mean([f["f1_curve"][k] for f in frs])), 4)
            for k in frs[0]["f1_curve"]}
        tp = sum(f["tp"] for f in frs)
        ng = sum(f["n_gt"] for f in frs)
        npd = sum(f["n_pred"] for f in frs)
        summary[f"inst_pooled{suf}"] = dict(
            tp=tp, fp=npd - tp, fn=ng - tp, n_gt=ng, n_pred=npd,
            f1_50=round(2 * tp / (npd + ng), 4) if (npd + ng) else 0.0)
        summary[f"n_gt_inst_mean{suf}"] = round(ng / len(frs), 2)
        summary[f"n_pred_inst_mean{suf}"] = round(npd / len(frs), 2)

    # per-frame の生値も保存（クリップ内分散・追跡可能性のため。フレーム順は abs_frames）
    summary["per_frame"] = [
        dict(abs_frame=af,
             mIoU=round(frame_mean(fr, "ious"), 4),
             mDice=round(frame_mean(fr, "dices"), 4),
             ious={str(k): round(v, 4) for k, v in fr["ious"].items()},
             dices={str(k): round(v, 4) for k, v in fr["dices"].items()},
             boundary_PRF=[round(x, 4) for x in fr["bf"]],
             boundary_PR_raw=[round(x, 4) for x in fr["br_raw"]],
             boundary_PRF_tol={str(t): [round(x, 4) for x in v]
                               for t, v in fr["bf_tol"].items()},
             boundary_PR_raw_tol={str(t): [round(x, 4) for x in v]
                                  for t, v in fr["br_raw_tol"].items()},
             underseg_error=round(fr["underseg"], 4),
             overseg={str(k): v for k, v in fr["overseg"].items()},
             n_regions=fr["n_regions"],
             inst=_inst_out(fr["inst"]),
             inst_labeled=_inst_out(fr["inst_dom"]["labeled"]),
             inst_tissue=_inst_out(fr["inst_dom"]["tissue"]),
             inst_labeled_tissue=_inst_out(fr["inst_dom"]["labeled_tissue"]),
             vi_split=_r(fr["vi_split"]) if np.isfinite(fr["vi_split"]) else None,
             vi_merge=_r(fr["vi_merge"]) if np.isfinite(fr["vi_merge"]) else None)
        for af, fr in zip(abs_frames_with_gt, per_frame)
    ]
    return summary


def evaluate_clip(root, clip, track_dir):
    """クリップ 1 本を評価して summary dict を返す（`main` の中身。テストからも呼ぶ）。

    Raises:
        SystemExit: ラベルが無い / フレーム番号が depth の範囲外 / GT を 1 枚も読めない。
    """
    depth = _load_depth(root, clip)
    N, H, W = depth.shape
    lab_paths = sorted(glob.glob(os.path.join(track_dir, "label_*.npy")))
    if not lab_paths:
        raise SystemExit(f"no labels in {track_dir}")

    labels, valids, abs_frames, gts = [], [], [], []
    per_frame = []
    for p in lab_paths:
        af = int(os.path.basename(p)[len("label_"):-len(".npy")])
        if not (0 <= af < N):                    # フレーム番号が depth 範囲外なら即失敗
            raise SystemExit(f"label frame {af} (from {os.path.basename(p)}) "
                             f"out of depth range [0,{N})")
        lab = np.load(p).astype(np.int32)        # cv2.resize は int64 を扱えないので int32
        if lab.shape != (H, W):                  # 念のため解像度を合わせる
            lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(depth[af]) & (depth[af] > 1e-6)
        gt = _gt_idmap(root, clip, af, H, W)
        labels.append(lab); valids.append(valid); abs_frames.append(af); gts.append(gt)
        if gt is not None:
            per_frame.append(eval_frame(lab, gt, valid))

    # GT を 1 枚も読めなかったら止める。放っておくと per_frame が空のまま全指標 0.0 を
    # 返し、「評価した結果ゼロ点」と区別が付かない（fail open）。このハーネスは GT との
    # 突き合わせが目的なので、GT ゼロは常に呼び出し側の誤りとして扱う。
    # 実際に踏んだ穴: cholec の `_color_mask.png` 決め打ちで ATLAS の `_class.png` が
    # 読めず、全フレーム GT 無し扱いになっていた。
    if not per_frame:
        seg_dir = os.path.join(root, clip, "seg_masks")
        why = ("seg_masks が無い（GT の無い tile クリップを渡していないか）"
               if not os.path.isdir(seg_dir) else
               "seg_masks はあるが 1 枚も読めなかった。対応形式は "
               "<frame>_color_mask.png (cholec) と <frame>_class.png (atlas)")
        raise SystemExit(f"GT が 1 枚も無いので評価できない: {seg_dir}\n  {why}\n"
                         f"  ラベルのフレーム番号: {abs_frames[:5]}…")

    with_gt = [a for a, g in zip(abs_frames, gts) if g is not None]
    return summarize(per_frame, labels, valids, with_gt)


def set_tissue_ignore(dataset: str | None, explicit: str = "") -> None:
    """`tissue` 土俵で外す GT クラス id を決める（`--dataset` か `--instrument_classes`）。

    両方渡されて食い違えば止める（黙ってどちらかを採らない）。
    """
    global TISSUE_IGNORE
    ex = tuple(int(x) for x in explicit.split(",") if x.strip() != "") if explicit else ()
    ds = INSTRUMENT_CLASSES.get(dataset) if dataset else None
    if dataset and ds is None:
        raise SystemExit(f"器具クラスが定義されていないデータセット: {dataset!r} "
                         f"（定義済み: {sorted(INSTRUMENT_CLASSES)}）")
    if ex and ds and tuple(sorted(ex)) != tuple(sorted(ds)):
        raise SystemExit(f"--instrument_classes {ex} が --dataset {dataset} の定義 {ds} と食い違う")
    TISSUE_IGNORE = ex or ds or ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../outputs/cholec_gt")
    ap.add_argument("--clip", default="VID01_s15_80_crop")
    ap.add_argument("--track_dir", required=True,
                    help="label_<absframe>.npy が並ぶディレクトリ。")
    ap.add_argument("--tag", default="run", help="出力 JSON 名/表示用ラベル。")
    ap.add_argument("--out_json", default="", help="指定時はここへ JSON 保存。")
    ap.add_argument("--extra_ignore", default="",
                    help="instance/VI の GT ドメインから追加除外するクラス id をカンマ区切りで"
                         "（例 '6,7'=connective/blood）。supplementary の GEOM 集計用。")
    ap.add_argument("--dataset", default="", choices=["", *sorted(INSTRUMENT_CLASSES)],
                    help="`tissue` 土俵で外す器具クラスの定義元。未指定なら tissue 系は出ない。")
    ap.add_argument("--instrument_classes", default="",
                    help="`tissue` 土俵で外す GT クラス id（カンマ区切り）。--dataset の代わり。")
    args = ap.parse_args()

    if args.extra_ignore.strip():
        global EXTRA_IGNORE
        EXTRA_IGNORE = {int(x) for x in args.extra_ignore.split(",") if x.strip() != ""}
    set_tissue_ignore(args.dataset or None, args.instrument_classes)

    summary = evaluate_clip(args.root, args.clip, args.track_dir)
    summary = dict(tag=args.tag, clip=args.clip, track_dir=args.track_dir, **summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
