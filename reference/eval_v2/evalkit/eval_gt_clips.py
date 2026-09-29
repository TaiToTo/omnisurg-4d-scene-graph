"""GT を持つクリップを一括評価して 1 枚の表にする入口（cholec / ATLAS 共通）。

`seg_masks/` を持つクリップを列挙し、1 本ずつ `eval_track.py` を回して平均を取る。
**評価セット（母集団）はここで固定する** — 軸ごとに違うクリップで測ると比較にならない
（悪いクリップだけ選ぶと改善しか出ない、という誤りを実際に踏んだ。
[../seg_quality_experiment/README.md](../seg_quality_experiment/README.md) §3）。

主指標は instance-F1 / PQ。領域数や最大塊は「潰れているか」の検出用で、品質指標では
ない（過分割を報酬にしてしまう。同 §1）。

出力 JSON には **どの評価コードで測ったか**（`eval_code_sha` / `eval_code_tag` /
`eval_code_files`）を残す。別バージョンの評価コードで測った条件同士を比べると
「手法の差」に「コードの差」が混ざるが、数字を見ても分からない。`compare_eval.py` は
`eval_code_sha` で突き合わせて止める（tag は人が読むためのラベルで、判定には使わない）。
sha はこのスクリプトと `eval_track.py` だけでなく、**GT のクラス id map を作る
surgical_core 側**（`FROZEN_GT_LOADERS`）まで含む — 指標を決めているのはそこも同じなので。

**F2-v3（2026-09-07）**: `eval_track.py` が F1-v2 になり、summary に v2 のキー
（SQ / 全閾値 F1 / 許容別 boundary / 土俵 4 種）が増えた。集計はこのファイルの
`SUMMARY_KEYS` が決める。`--dataset` を渡す（`--root` の末尾が `atlas` / `cholec_gt` /
`lapex` なら推定する）と `tissue` 土俵の器具クラスが決まる。推定できなければ止まる。

`--dataset` は**同じ評価コードのまま `tissue` 土俵の中身を変える**（推定と明示で違う値に
なりうる）ので、`check_comparable` は sha と母集団だけでなく**評価ドメイン
（`dataset` / `tissue_ignore` / `extra_ignore`）の一致まで見る**。

Usage:
    # ATLAS（既定）
    python3 depth_sam_tracking_experiment/eval_gt_clips.py \\
        --root outputs/atlas --sam-out out_atlas --track-dir-name track_rgb --tag baseline
    # cholec
    python3 depth_sam_tracking_experiment/eval_gt_clips.py \\
        --root outputs/cholec_gt --sam-out out --track-dir-name track_rgb --tag cholec_base
    # 対比較は --clips で母集団を固定する（両方に同じリストを渡す）
    python3 ... --track-dir-name track_rgb_e3 --tag cons5 --clips clipA,clipB

旧名 `seg_quality_experiment/scripts/eval_atlas_gt_clips.py` は後方互換の shim。
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(EXP)

# 凍結の単位。この内容が変われば sha が変わり、過去の条件と比較できなくなる。
# git SHA ではなく**ファイルの内容**にするのは、dirty な作業ツリーでも正しく効かせるため。
FROZEN_FILES = ("eval_track.py", "eval_gt_clips.py")

# 指標を決めているのはスクリプト 2 本だけではない。GT のクラス id map を作るのは
# surgical_core 側で、ここを差し替えると全指標が変わるのにスクリプトの sha は動かない
# （＝別物で測った条件を `compare_eval.py` が素通しする）。定義元のファイルまで sha に含める。
FROZEN_GT_LOADERS = (("surgical_core.cholec", "mask_to_id_map"),
                     ("surgical_core.atlas", "load_atlas_id_map"))

# `--root` の末尾からデータセット名を推定する表。ここに無い root は `--dataset` 必須。
DATASET_OF_ROOT = {"atlas": "atlas", "cholec_gt": "cholec", "lapex": "lapex"}

# summary に載せるクリップ平均のキー。(キー, None を許すか)。
# 先頭 8 つは F2-v2 と同じ並び（表示順もこの順）。
SUMMARY_KEYS = (
    ("inst_F1_50", False), ("inst_F1_75", False), ("PQ", False), ("GT_mIoU", False),
    ("underseg_error", False), ("overseg_mean", False), ("n_regions_mean", False),
    ("time_IoU", False),
    # ── v2 ──
    ("inst_F1_avg", False), ("SQ", True), ("inst_BF", True),
    ("GT_mDice", False), ("boundary_F", False), ("boundary_P", False), ("boundary_R", False),
    ("boundary_P_raw", False), ("boundary_R_raw", False),
    ("boundary_F_tol1", False), ("boundary_F_tol3", False), ("boundary_F_tol5", False),
    ("boundary_R_raw_tol1", False), ("boundary_R_raw_tol3", False), ("boundary_R_raw_tol5", False),
    ("boundary_P_raw_tol1", False), ("boundary_P_raw_tol3", False), ("boundary_P_raw_tol5", False),
    ("VI_split", True), ("VI_merge", True),
    ("inst_F1_50_labeled", True), ("inst_F1_75_labeled", True), ("inst_F1_avg_labeled", True),
    ("PQ_labeled", True), ("SQ_labeled", True), ("inst_BF_labeled", True),
    ("inst_F1_50_tissue", True), ("inst_F1_75_tissue", True), ("inst_F1_avg_tissue", True),
    ("PQ_tissue", True), ("SQ_tissue", True), ("inst_BF_tissue", True),
    ("inst_F1_50_labeled_tissue", True), ("inst_F1_75_labeled_tissue", True),
    ("inst_F1_avg_labeled_tissue", True), ("PQ_labeled_tissue", True),
    ("SQ_labeled_tissue", True), ("inst_BF_labeled_tissue", True),
    ("n_gt_inst_mean", False), ("n_pred_inst_mean", False),
)


def dataset_of(root: str, explicit: str = "") -> str:
    """`--dataset` が無ければ root の末尾から推定する。どちらも無ければ止める（fail closed）。"""
    if explicit:
        return explicit
    ds = DATASET_OF_ROOT.get(os.path.basename(os.path.normpath(root)))
    if ds is None:
        raise SystemExit(f"--root {root!r} からデータセットを推定できない。--dataset を渡すこと"
                         f"（{sorted(DATASET_OF_ROOT.values())}）")
    return ds


# `eval_track.py` の `from surgical_core... import ...` を拾う。カバー漏れの検査用。
_SC_IMPORT_RE = re.compile(r"^from\s+(surgical_core[\w.]*)\s+import\s+(.+)$", re.M)


def _check_gt_loaders_covered() -> None:
    """`eval_track.py` の surgical_core 依存が全部 sha に入っているか検査する（fail closed）。

    依存を足したのに `FROZEN_GT_LOADERS` を直し忘れると、**指標が変わったのに sha が
    動かない**状態へ黙って戻る。それは凍結の意味が無くなる不変条件の破れなので raise する。

    Raises:
        RuntimeError: `eval_track.py` が sha に入っていない surgical_core の名前を import している。
    """
    with open(os.path.join(EXP, "eval_track.py"), encoding="utf-8") as f:
        src = f.read()
    imported = set()
    for mod, names in _SC_IMPORT_RE.findall(src):
        for n in names.split("#")[0].split(","):
            n = n.strip().split(" as ")[0].strip()
            if n:
                imported.add((mod, n))
    missing = sorted(imported - set(FROZEN_GT_LOADERS))
    if missing:
        raise RuntimeError(
            "eval_track.py が sha に入っていない surgical_core の名前を import している。\n"
            f"  未カバー: {missing}\n"
            "  そのまま測ると『指標が変わったのに eval_code_sha は同じ』になり、\n"
            "  違う物差しの条件を compare_eval.py が素通しする。\n"
            "  eval_gt_clips.FROZEN_GT_LOADERS に足し、凍結タグを打ち直すこと。")


def frozen_parts() -> list[tuple[str, str, bytes]]:
    """sha の材料を `(ラベル, 解決したパス, 内容)` で返す。順序は固定。

    surgical_core 側は**実際に import されたモジュール**からソースを引く。パスは sha に
    混ぜない（同じ内容なら測定は同じなので、クローン位置で sha が動かない方が正しい）が、
    「どのファイルで測ったか」を JSON に残すために返り値には含める。
    """
    parts, seen = [], set()
    for name in FROZEN_FILES:
        path = os.path.join(EXP, name)
        with open(path, "rb") as f:
            parts.append((name, path, f.read()))
        seen.add(path)
    for mod_name, func_name in FROZEN_GT_LOADERS:
        obj = getattr(importlib.import_module(mod_name), func_name)
        path = inspect.getsourcefile(obj)
        if path is None:                         # C 拡張などソースを引けないものは凍結できない
            raise RuntimeError(f"{mod_name}.{func_name} のソースが引けないので凍結できない")
        if path in seen:                         # 同じファイルに 2 つ定義されていても 1 回だけ
            continue
        with open(path, "rb") as f:
            parts.append((f"{mod_name}.{func_name}", path, f.read()))
        seen.add(path)
    return parts


def eval_code_sha() -> str:
    """評価コードの内容 sha256（`FROZEN_FILES` + `FROZEN_GT_LOADERS` の定義元）。"""
    _check_gt_loaders_covered()
    h = hashlib.sha256()
    for label, _path, content in frozen_parts():
        h.update(label.encode("utf-8"))
        h.update(b"\0")
        h.update(content)
    return h.hexdigest()


def eval_code_files() -> list[str]:
    """sha に入っているファイル（REPO 相対。どこで解決されたかを JSON に残す用）。"""
    return [os.path.relpath(path, REPO) for _label, path, _c in frozen_parts()]


def eval_code_tag() -> str | None:
    """凍結タグ名（環境変数 `IPCAI_EVAL_TAG`）。未設定なら None。"""
    return os.environ.get("IPCAI_EVAL_TAG") or None


def gt_clips(root: str) -> list[str]:
    """GT を持つクリップ名を並べる（`seg_masks/` があるもの）。"""
    return sorted(c for c in os.listdir(root)
                  if os.path.isdir(os.path.join(root, c, "seg_masks")))


def clips_of(summary: dict) -> list[str]:
    """評価 JSON から母集団を取り出す（`clips` が無い古い JSON は per_clip から復元）。"""
    return list(summary.get("clips") or [r["clip"] for r in summary["per_clip"]])


def eval_domain(summary: dict) -> dict | None:
    """`tissue` 土俵を決めている設定を取り出す（F1-v2 の JSON だけが持つ）。

    同じ `eval_code_sha` でも `--dataset` / `--extra_ignore` は実行時に決まるので、
    **コードが同じでも評価ドメインは違いうる**（`--root` からの推定と明示指定で
    値が割れる、など）。ここが違う 2 条件を並べると `*_tissue` の指標が別物なのに
    数字だけは揃って見える。

    Args:
        summary: 評価 JSON（`eval_gt_clips.py` が書いたもの）。
    Returns:
        dict(dataset, tissue_ignore, extra_ignore)。v1 の JSON（`eval_version` なし）は None。
    """
    if summary.get("eval_version") is None:
        return None
    # extra_ignore は per_clip にしか無い（クリップ間で割れていれば見えるように集合で持つ）。
    extra = sorted({tuple(sorted(r.get("extra_ignore") or ()))
                    for r in summary["per_clip"]})
    return dict(dataset=summary.get("dataset"),
                tissue_ignore=sorted(summary.get("tissue_ignore") or ()),
                extra_ignore=[list(x) for x in extra])


def check_comparable(a: dict, b: dict, allow_subset: bool = False,
                     allow_legacy_code: bool = False) -> dict:
    """2 つの評価 JSON が「同じ物差し・同じ母集団」か検査する。

    どちらの検査も**警告ではなく raise** で止める。混ざったことに気づかないのが
    一番怖い（ATLAS の「本番の現状」は単一 seed 13 本 + consensus 5 本の混合で、
    新条件を 18 本で回して比べると手法混合との比較になっていた）。

    Args:
        a: 基準側の評価 JSON（`eval_gt_clips.py` が書いたもの）。
        b: 比較側の評価 JSON。
        allow_subset: True なら母集団が違っても共通部分で比較する。
        allow_legacy_code: True なら `eval_code_sha` を持たない古い JSON を通す。

    Returns:
        dict(clips=比較に使うクリップ（ソート済み）, population="identical"|"intersection",
             eval_code="<sha16>"|"legacy-unverified")。

    Raises:
        ValueError: 評価コードが食い違う / 記録が無い / **評価ドメインが違う** /
            母集団が違う / 共通部分が空。
    """
    sa, sb = a.get("eval_code_sha"), b.get("eval_code_sha")
    ta, tb = a.get("eval_code_tag"), b.get("eval_code_tag")
    if sa is None or sb is None:
        if not allow_legacy_code:
            raise ValueError(
                "評価コードの記録が無い JSON がある（eval_code_sha なし）。どの物差しで\n"
                f"  測ったか復元できないので比較できない: base={sa!r} cond={sb!r}\n"
                "  eval_gt_clips.py で測り直すか、承知の上なら --allow-legacy-code を付ける\n"
                "  （summary に eval_code: legacy-unverified が残る）")
        code = "legacy-unverified"
    elif sa != sb:
        raise ValueError(
            "評価コードが違う 2 条件は比較できない（手法の差にコードの差が混ざる）。\n"
            f"  base: tag={ta!r} sha={sa[:16]}\n"
            f"  cond: tag={tb!r} sha={sb[:16]}\n"
            "  どちらかを同じ評価コードで測り直すこと（凍結タグは "
            "docs/paper/ipcai2027_work_plan.md §3）")
    else:
        code = sa[:16]

    # 評価コードが同じでも `tissue` 土俵は実行時引数で変わる。片方が v1 の JSON なら
    # 比べる相手が無いので飛ばす（sha が一致していれば両方 v2 なので、実質
    # `--allow-legacy-code` のときだけ起きる）。
    da, db = eval_domain(a), eval_domain(b)
    if da is not None and db is not None and da != db:
        raise ValueError(
            "評価ドメインが違う 2 条件は比較できない（`*_tissue` の土俵が別物になる）。\n"
            f"  base: {da}\n  cond: {db}\n"
            "  同じ --dataset / --extra_ignore で測り直すこと"
            "（--dataset は --root の末尾から推定もされるので、明示指定と食い違いうる）")

    ca, cb = clips_of(a), clips_of(b)
    if sorted(ca) == sorted(cb):
        return dict(clips=sorted(ca), population="identical", eval_code=code)

    only_a, only_b = sorted(set(ca) - set(cb)), sorted(set(cb) - set(ca))
    if not allow_subset:
        raise ValueError(
            "クリップ集合が違うので比較できない（母集団の違う平均を並べてはいけない）。\n"
            f"  base だけ: {only_a}\n  cond だけ: {only_b}\n"
            "  同じ --clips を渡して測り直すか、承知の上なら --allow-subset を付ける")
    common = sorted(set(ca) & set(cb))
    if not common:
        raise ValueError(f"共通クリップが 0 なので --allow-subset でも比較できない。"
                         f"base={len(ca)} 本 / cond={len(cb)} 本")
    return dict(clips=common, population="intersection", eval_code=code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/atlas", help="クリップのルート（REPO 相対）。")
    ap.add_argument("--sam-out", default="out_atlas", help="追跡結果のルート（EXP 相対）。")
    ap.add_argument("--track-dir-name", default="track_rgb",
                    help="クリップ内のラベル置き場。条件ごとに変える。")
    ap.add_argument("--tag", default="baseline", help="この条件の名前。")
    ap.add_argument("--clips", default="",
                    help="評価するクリップをカンマ区切りで固定する（既定は GT のある全部）。"
                         "条件間で母集団が変わると平均が比較できないので、対比較では必ず指定する。"
                         "指定したクリップにラベルが無ければ **止まる**（黙って母集団が縮むのを防ぐ）。")
    ap.add_argument("--out", default="ipcai2027_experiment/out/eval",
                    help="JSON の出力先（REPO 相対）。")
    ap.add_argument("--dataset", default="",
                    help="`tissue` 土俵で外す器具クラスの定義元（atlas / cholec / lapex）。"
                         "省略時は --root の末尾から推定し、推定できなければ止まる。")
    args = ap.parse_args()
    dataset = dataset_of(args.root, args.dataset)

    root_abs = os.path.join(REPO, args.root)
    out_dir = os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{os.path.join(REPO, 'sam3_wrapper')}")

    clips = gt_clips(root_abs)
    if args.clips:                               # 母集団を固定するときは存在確認まで厳密に
        want = [c.strip() for c in args.clips.split(",") if c.strip()]
        unknown = [c for c in want if c not in clips]
        if unknown:
            raise SystemExit(f"GT を持たない / 存在しないクリップを指定した: {unknown}")
        clips = want

    rows, missing, failed = [], [], []
    for clip in clips:
        track_dir = os.path.join(args.sam_out, clip, args.track_dir_name)
        if not os.path.isdir(os.path.join(EXP, track_dir)):
            if args.clips:                       # 明示指定なら黙って減らさず止まる
                raise SystemExit(f"{clip} に {args.track_dir_name} が無い。"
                                 "母集団を固定した評価なので中断する")
            missing.append(clip)
            continue
        r = subprocess.run(
            [sys.executable, "eval_track.py",
             "--root", os.path.relpath(root_abs, EXP), "--clip", clip,
             "--track_dir", track_dir, "--tag", args.tag, "--dataset", dataset],
            cwd=EXP, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            why = (r.stderr.strip().splitlines()[-1] if r.stderr and r.stderr.strip()
                   else f"exit {r.returncode}")
            # ラベルが無いときと同じ扱い。母集団を固定した評価は 1 本欠けた時点で
            # 「指定した母集団の平均」ではなくなるので、続けずに止める。
            if args.clips:
                raise SystemExit(f"{clip} の評価が失敗した: {why}\n"
                                 "母集団を固定した評価なので中断する")
            print(f"  [FAIL] {clip}: {why}", file=sys.stderr)
            failed.append(clip)
            continue
        # eval_track.py は summary + per_frame の JSON を stdout に出す
        rows.append(json.loads(r.stdout))
        print(f"  {clip:44s} F1@.5={rows[-1]['inst_F1_50']:.3f} PQ={rows[-1]['PQ']:.3f} "
              f"bF={rows[-1]['boundary_F']:.3f}", flush=True)

    if missing:
        print(f"\n[WARN] {len(missing)} clips に {args.track_dir_name} が無い: "
              f"{missing[:3]}…", file=sys.stderr)
    if failed:
        print(f"[WARN] {len(failed)} clips の評価が失敗した（母集団が縮んでいる）: "
              f"{failed[:3]}…", file=sys.stderr)
    if not rows:
        raise SystemExit(f"評価できたクリップが 0。{args.track_dir_name} を作ってから回すこと")

    def mean(k, allow_none=False):
        vals = [r[k] for r in rows]
        if allow_none:                          # 土俵が定義できないクリップは平均から外す
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        if any(v is None for v in vals):
            raise SystemExit(f"{k}: None を含む（土俵が定義できないクリップがある）。"
                             "allow_none でないキーなので止める")
        return round(sum(vals) / len(rows), 4)

    summary = {
        "tag": args.tag, "track_dir_name": args.track_dir_name,
        # どの評価コードで測ったか。これが無いと条件間の比較が正しいか検証できない。
        # eval_code_files は sha の中身（凍結の単位）を JSON だけで復元できるようにする。
        "eval_code_sha": eval_code_sha(), "eval_code_tag": eval_code_tag(),
        "eval_code_files": eval_code_files(),
        "eval_version": rows[0].get("eval_version"),
        "root": args.root, "dataset": dataset,
        "tissue_ignore": rows[0].get("tissue_ignore"),
        # 母集団を JSON に残す。後から「どのクリップで測った数字か」を復元できないと、
        # 条件間の比較が正しいかどうか検証できない。
        "clips": [r["clip"] for r in rows],
        "n_clips": len(rows), "n_missing": len(missing), "n_failed": len(failed),
    }
    for k, allow_none in SUMMARY_KEYS:
        summary[k] = mean(k, allow_none)
    # 土俵ごとに「平均に入ったクリップ数」を残す（None を外した平均は母集団が縮む）
    for suf in ("_labeled", "_tissue", "_labeled_tissue"):
        summary[f"n_clips_inst{suf}"] = sum(
            1 for r in rows if r.get(f"inst_F1_50{suf}") is not None)
    # **`None` を外した平均には、その指標自身の本数を添える。**（監査 R1 の書き手側。）
    # `n_clips_inst{suf}` は `inst_F1_50{suf}` の本数であって SQ / inst_BF の本数ではない
    # — こちらは「マッチが 1 つも無い（TP=0）」でも落ちるのでもっと狭く、実測で 124 条件中
    # 56 条件がずれる。本数を書き分けないと、読む側は縮んだ母集団の平均を
    # そうと気づかずに引き算する（それが R1 そのもの）。
    for k, allow_none in SUMMARY_KEYS:
        if allow_none:
            summary[f"n_clips_{k}"] = sum(1 for r in rows if r.get(k) is not None)
    summary["per_clip"] = rows

    path = os.path.join(out_dir, f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\n=== {args.tag} ({len(rows)} clips, {dataset}) ===")
    for k in ("inst_F1_50", "inst_F1_75", "inst_F1_avg", "PQ", "SQ", "GT_mIoU",
              "boundary_F", "boundary_R_raw", "boundary_P_raw", "underseg_error",
              "overseg_mean", "n_regions_mean", "time_IoU",
              "inst_F1_50_labeled", "inst_F1_50_tissue", "inst_F1_50_labeled_tissue"):
        print(f"  {k:26s} {summary[k]}")
    print(f"  eval_code       {summary['eval_code_tag'] or '(no tag)'} "
          f"{summary['eval_code_sha'][:16]}")
    print(f"→ {path}")


if __name__ == "__main__":
    main()
