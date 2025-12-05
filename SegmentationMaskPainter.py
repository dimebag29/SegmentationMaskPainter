# ==============================================================================================================
# 作成者:dimebag29 作成日:2025年12月5日 バージョン:v0.0
# (Author:dimebag29 Creation date:December 5, 2025 Version:v0.0)
#
# このプログラムは大部分をAI (Gemini 3.0 Pro, ChatGPT 5.1)を利用して作成されました。
# (This program was created largely using AI (Gemini 3.0 Pro, ChatGPT 5.1). )
#
# このプログラムのライセンスはCC0 (クリエイティブ・コモンズ・ゼロ)です。いかなる権利も保有しません。
# (This program is licensed under CC0 (Creative Commons Zero). No rights reserved.)
# https://creativecommons.org/publicdomain/zero/1.0/
#
# 開発環境:
# ･Windows 10 64bit, AMD Ryzen 7 5700X, NVIDIA GeForce RTX 3090 - Driver 571.96 - CUDA 12.8
# ･python 3.12.0
# ･pyinstaller 6.17.0
# ･auto-py-to-exe 2.48.1 (pyinstallerをGUI化する便利ライブラリ)
#
# exe化時のauto-py-to-exeの設定:
# ･ひとつのファイルにまとめる (--onefile)
# ･ウィンドウベース (--windowed)
# ･exeアイコン設定 (--icon)
# ･高度な設定でscipyを同梱 (--collect-all scipy) ※exe化後、処理実行時に要求するエラーが出る
#
# ==============================================================================================================
# exe化時のバグ 1:
# 「NameError: name 'name' is not defined」というエラーが出てexeが起動できない。
# 以下のサイトを参考にAppData\Local\Programs\Python\Python312\Lib\site-packages\torch\_numpy\_ufuncs.pyを編集する必要がある。
# https://stackoverflow.com/questions/78375284/torch-error-nameerror-name-name-is-not-defined
# 編集するのは以下の2か所
"""
1.
for name in _binary:
    ufunc = getattr(_binary_ufuncs_impl, name)
    ufunc_name = name  
    vars()[ufunc_name] = deco_binary_ufunc(ufunc)
2.
for name in _unary:
    ufunc = getattr(_unary_ufuncs_impl, name)
    #vars()[name] = deco_unary_ufunc(ufunc)
    ufunc_name = name  # Définir une variable avec le nom de l'ufunc
    vars()[ufunc_name] = deco_binary_ufunc(ufunc)
"""
#
# exe化時のバグ 2:
#「NameError: name 'obj' is not defined」というエラーが出てexeが起動できない。
# AppData\Local\Programs\Python\Python312\Lib\site-packages\scipy\stats\_distn_infrastructure.pyの369行目
# 「del obj」を「#del obj」とコメントアウトする
# ==============================================================================================================

import os
import sys
import time
from datetime import datetime, timedelta
import glob
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch                                                                        # 2.9.0+cu128 (対応アーキテクチャ: sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120)
import numpy as np                                                                  # 2.1.2
from PIL import Image                                                               # 11.3.0
from scipy.ndimage import binary_dilation                                           # 1.16.1
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation    # 4.57.3



# =========================================================
#  クラス名データ（日本語, 英語）
# =========================================================
CLASS_DATA = [
    ("壁", "wall"), ("建物", "building"), ("空", "sky"), ("床", "floor"), ("木", "tree"),
    ("天井", "ceiling"), ("道路", "road"), ("ベッド", "bed"), ("窓ガラス", "windowpane"), ("草", "grass"),
    ("キャビネット", "cabinet"), ("歩道", "sidewalk"), ("人", "person"), ("地面（土）", "earth"), ("ドア", "door"),
    ("テーブル", "table"), ("山", "mountain"), ("植物", "plant"), ("カーテン", "curtain"), ("椅子", "chair"),
    ("車", "car"), ("水", "water"), ("絵画", "painting"), ("ソファ", "sofa"), ("棚", "shelf"),
    ("家", "house"), ("海", "sea"), ("鏡", "mirror"), ("ラグ（敷物）", "rug"), ("野原", "field"),
    ("アームチェア（ひじ掛け椅子）", "armchair"), ("席", "seat"), ("フェンス（柵）", "fence"), ("机", "desk"), ("岩", "rock"),
    ("ワードローブ（洋服ダンス）", "wardrobe"), ("ランプ", "lamp"), ("浴槽", "bathtub"), ("手すり", "railing"), ("クッション", "cushion"),
    ("台座・基部", "base"), ("箱", "box"), ("柱", "column"), ("看板", "signboard"), ("引き出し付きタンス", "chest of drawers"),
    ("カウンター", "counter"), ("砂", "sand"), ("流し台（シンク）", "sink"), ("超高層ビル", "skyscraper"), ("暖炉", "fireplace"),
    ("冷蔵庫", "refrigerator"), ("スタンド（観覧席）", "grandstand"), ("小道", "path"), ("階段", "stairs"), ("滑走路", "runway"),
    ("ケース", "case"), ("ビリヤード台", "pool table"), ("枕", "pillow"), ("網戸（スクリーンドア）", "screen door"), ("階段（階段通路）", "stairway"),
    ("川", "river"), ("橋", "bridge"), ("本棚", "bookcase"), ("ブラインド", "blind"), ("コーヒーテーブル", "coffee table"),
    ("トイレ", "toilet"), ("花", "flower"), ("本", "book"), ("丘", "hill"), ("ベンチ", "bench"),
    ("カウンタートップ", "countertop"), ("コンロ（ストーブ）", "stove"), ("ヤシの木", "palm"), ("キッチンアイランド", "kitchen island"), ("コンピュータ", "computer"),
    ("回転椅子", "swivel chair"), ("ボート", "boat"), ("バー（酒場）", "bar"), ("アーケードゲーム機", "arcade machine"), ("掘っ立て小屋", "hovel"),
    ("バス", "bus"), ("タオル", "towel"), ("明かり（照明）", "light"), ("トラック", "truck"), ("塔", "tower"),
    ("シャンデリア", "chandelier"), ("オーニング（日よけ）", "awning"), ("街灯", "streetlight"), ("ブース（屋台・小室）", "booth"), ("テレビ受像機", "television receiver"),
    ("飛行機", "airplane"), ("ダートトラック（未舗装路）", "dirt track"), ("衣類", "apparel"), ("杭・ポール", "pole"), ("陸地", "land"),
    ("手すり（階段などの）", "bannister"), ("エスカレーター", "escalator"), ("オットマン（足置き）", "ottoman"), ("ボトル", "bottle"), ("ビュッフェ台（配膳台）", "buffet"),
    ("ポスター", "poster"), ("舞台", "stage"), ("バン（車）", "van"), ("船", "ship"), ("噴水", "fountain"),
    ("コンベヤーベルト", "conveyer belt"), ("天蓋（キャノピー）", "canopy"), ("洗濯機", "washer"), ("おもちゃ", "plaything"), ("プール", "swimming pool"),
    ("スツール（背もたれなし椅子）", "stool"), ("樽", "barrel"), ("バスケット", "basket"), ("滝", "waterfall"), ("テント", "tent"),
    ("バッグ", "bag"), ("ミニバイク", "minibike"), ("揺りかご", "cradle"), ("オーブン", "oven"), ("ボール", "ball"),
    ("食べ物", "food"), ("段・ステップ", "step"), ("タンク", "tank"), ("商標（ブランド名）", "trade name"), ("電子レンジ", "microwave"),
    ("鍋", "pot"), ("動物", "animal"), ("自転車", "bicycle"), ("湖", "lake"), ("食器洗い機", "dishwasher"),
    ("画面（スクリーン）", "screen"), ("毛布", "blanket"), ("彫刻", "sculpture"), ("フード（換気フード）", "hood"), ("壁灯（ブラケット）", "sconce"),
    ("花瓶", "vase"), ("信号機", "traffic light"), ("トレイ", "tray"), ("ごみ箱", "ashcan"), ("扇風機", "fan"),
    ("桟橋", "pier"), ("CRT画面（ブラウン管）", "crt screen"), ("皿", "plate"), ("モニター", "monitor"), ("掲示板", "bulletin board"),
    ("シャワー", "shower"), ("ラジエーター（暖房器）", "radiator"), ("ガラス", "glass"), ("時計", "clock"), ("旗", "flag")
]

# =========================================================
#  デフォルト設定値 (設定ファイルがない場合に使用)
# =========================================================
DEFAULT_TARGETS = [
    ["person", "0", "0", "0", "0", "5"],
    ["bag", "0", "0", "0", "0", "5"],
    ["car", "0", "0", "0", "0", "5"],
    ["truck", "0", "0", "0", "0", "5"],
    ["bus", "0", "0", "0", "0", "5"],
    ["sky", "67", "148", "240", "255", "1"]
]

# =========================================================
#  保存フォルダ（AppData\Local）
# =========================================================
def get_appdata_dir():
    # 実行ファイルまたはスクリプトのあるディレクトリを使用
    path = os.path.dirname(sys.argv[0])
    os.makedirs(path, exist_ok=True)
    return path

SETTINGS_PATH = os.path.join(get_appdata_dir(), "settings.json")


# =========================================================
#  設定セーブ / ロード
# =========================================================
def save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"設定を保存しました: {SETTINGS_PATH}")
    except Exception as e:
        print(f"設定保存エラー: {e}")


def load_settings():
    if not os.path.exists(SETTINGS_PATH):
        return None

    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


# =========================================================
#  print → GUI Text
# =========================================================
class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.config(state="normal")
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)
        self.widget.config(state="disabled")

    def flush(self):
        pass


# =========================================================
#  セグメンテーション処理
# =========================================================
def load_model(inf_short, inf_long):
    print("データセットを読み込んでいます")

    model_name = "facebook/mask2former-swin-large-ade-semantic"
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"shortest_edge": inf_short, "longest_edge": inf_long},
        )
        # ユーザー指定: Mask2FormerForUniversalSegmentation を使用
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        model.eval()

        if torch.cuda.is_available():
            model.to("cuda")
            print("CUDAが利用可能。GPUを使用します")
        else:
            print("CUDAが利用不可。CPUを使用します")

        return processor, model

    except Exception as e:
        print(f"データセットロードエラー: {e}")
        return None, None


def apply_segmentation(image_path, processor, model, output_dir, TARGET_COLORS):
    try:
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        save_path = os.path.join(output_dir, name + ".png")

        if os.path.exists(save_path):
            print(f"既にファイルが存在したのでスキップ: {filename}")
            return False

        original_image = Image.open(image_path).convert("RGBA")
        image_input = original_image.convert("RGB")

        # 入力をモデルと同じデバイスに送る
        device = model.device
        inputs = processor(images=image_input, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # === 修正箇所: GPUカーネルエラー発生時のフォールバック処理 ===
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        except RuntimeError as e:
            # 特定のCUDAエラーをキャッチ
            if "no kernel image is available" in str(e) or "CUDA" in str(e):
                print(f"!!! GPUエラー検出: {e}")
                print("!!! GPUがこのモデルに対応していない可能性があるため、CPUモードに切り替えて再試行します...")
                
                # モデルをCPUに移動
                model.to("cpu")
                # 入力データもCPUに移動
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                
                # CPUで再実行
                with torch.no_grad():
                    outputs = model(**inputs)
            else:
                # それ以外のエラーはそのまま投げる
                raise e
        # =======================================================

        pred_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image_input.size[::-1]]
        )[0]

        # 結果をCPUに戻してnumpy化
        pred_np = pred_map.cpu().numpy()

        # マスク画像を作成。RGBAの4チャンネルで初期化
        H, W = original_image.size[::-1]
        
        # ターゲット色のための配列
        mask_arr = np.zeros((H, W, 4), dtype=np.uint8)
        
        # どのピクセルがターゲットとして検出されたかを記録するマスク (Trueなら上書き対象)
        found_mask = np.zeros((H, W), dtype=bool)

        id2label = getattr(model.config, "id2label", {i: f"class_{i}" for i in range(150)})

        unique_labels = np.unique(pred_np)
        found_targets = []

        for label_id in unique_labels:
            if label_id not in id2label:
                continue

            name_lbl = id2label[label_id].lower()

            if name_lbl not in TARGET_COLORS:
                continue

            found_targets.append(name_lbl)

            r, g, b, a, exp = TARGET_COLORS[name_lbl]

            mask = pred_np == label_id

            if exp > 0:
                # 拡張処理
                structure = np.ones((2 * exp + 1, 2 * exp + 1), dtype=bool)
                mask = binary_dilation(mask, structure=structure)

            # マスクがTrueのピクセルにTARGET_COLORSを設定
            mask_arr[mask] = (r, g, b, a)
            
            # このピクセルは上書き対象としてマークする
            found_mask[mask] = True

        # -------------------------------------------------------------
        # 強制的な塗りつぶしロジック
        # -------------------------------------------------------------
        # 元画像をnumpy配列化
        final_np = np.array(original_image, dtype=np.uint8)
        
        # ターゲットとして検出された全てのピクセル(found_maskがTrueの場所)を
        # mask_arrの設定色(RGBA)で完全に置換する。
        final_np[found_mask] = mask_arr[found_mask]
        
        # 画像として保存
        merged_image = Image.fromarray(final_np, "RGBA")
        merged_image.save(save_path)

        print(f"完了: {filename}")
        return True

    except Exception as e:
        print(f"エラー({image_path}): {e}")
        import traceback
        traceback.print_exc()
        return False


# =========================================================
#  全画像処理 (中断機能付き)
# =========================================================
def start_processing(input_folder, output_folder, target_colors, inf_short, inf_long, stop_event):
    if not os.path.exists(input_folder):
        print("入力フォルダが存在しません")
        return

    os.makedirs(output_folder, exist_ok=True)

    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not files:
        print("画像が見つかりません")
        return

    print(f"推論解像度 shortest={inf_short}[px], longest={inf_long}[px]")

    processor, model = load_model(inf_short, inf_long)
    if processor is None:
        return

    total = len(files)
    print(f"{total} 枚の画像を処理します")
    print("")

    start = time.time()

    for i, f in enumerate(files):
        # 中断チェック
        if stop_event.is_set():
            print(">>> ユーザー操作により処理を中断しました <<<\n")
            break

        apply_segmentation(f, processor, model, output_folder, target_colors)

        elapsed = time.time() - start
        if i + 1 > 0:
            eta = datetime.now() + timedelta(seconds=(elapsed / (i + 1)) * (total - (i + 1)))
            print(f"{round((i+1)/total * 100.0, 1)}% [{i+1}/{total}], 終了予定時刻: {eta.strftime('%H:%M:%S')}")

    if not stop_event.is_set():
        print("\n■■■■■ 全処理完了!! ■■■■■\n")


# =========================================================
#  GUI 本体
# =========================================================
class SegGUI:
    def __init__(self, root):
        self.root = root
        root.title("SegmentationMaskPainter v0.0")
        
        # 中断制御用のイベント
        self.stop_event = threading.Event()

        # ==============================
        # 入出力フォルダ
        # ==============================
        frm_io = ttk.LabelFrame(root, text="フォルダ設定")
        frm_io.pack(fill="x", padx=10, pady=5)

        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()

        ttk.Label(frm_io, text="Input Folder ").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_io, textvariable=self.in_var, width=50).grid(row=0, column=1)
        ttk.Button(frm_io, text="参照", command=self.sel_in).grid(row=0, column=2)

        ttk.Label(frm_io, text="Output Folder ").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_io, textvariable=self.out_var, width=50).grid(row=1, column=1)
        ttk.Button(frm_io, text="参照", command=self.sel_out).grid(row=1, column=2)

        # ==============================
        # 解像度
        # ==============================
        frm_res = ttk.LabelFrame(root, text="推論解像度設定")
        frm_res.pack(fill="x", padx=10, pady=5)

        self.short_var = tk.IntVar(value=1024)
        self.long_var = tk.IntVar(value=2048)

        ttk.Label(frm_res, text="Shortest [px] (初期値:1024) ").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_res, textvariable=self.short_var, width=10).grid(row=0, column=1)

        ttk.Label(frm_res, text="Longest [px] (初期値:2048) ").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_res, textvariable=self.long_var, width=10).grid(row=1, column=1)

        # ==============================
        # Excel風ターゲットテーブル
        # ==============================
        frm_tbl = ttk.LabelFrame(root, text="マスク設定")
        frm_tbl.pack(fill="both", padx=10, pady=5, expand=True)

        # ---- class一覧ボタン用フレーム ----
        frm_btn = tk.Frame(frm_tbl)
        frm_btn.pack(side="top", fill="x", padx=5, pady=2)
        
        # class一覧ボタン (左側に配置)
        btn_class_list = ttk.Button(frm_btn, text="対象物一覧", command=self.open_class_list)
        btn_class_list.pack(side="left")

        # キャンバスとスクロールバー
        canvas = tk.Canvas(frm_tbl, height=300)
        scrollbar = ttk.Scrollbar(frm_tbl, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.table_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.table_frame, anchor="nw")

        headers = ["対象物", "R (赤)", "G (緑)", "B (青)", "A (透明)", "マスク拡張[px]"]
        for col, h in enumerate(headers):
            tk.Label(self.table_frame, text=h, borderwidth=1, relief="solid", width=12).grid(row=0, column=col)

        self.cells = []
        for r in range(1, 151):
            row_cells = []
            for c in range(6):
                e = tk.Entry(self.table_frame, width=12)
                e.grid(row=r, column=c)
                row_cells.append(e)
            self.cells.append(row_cells)

        self.table_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # ==============================
        # ログ
        # ==============================
        frm_log = ttk.LabelFrame(root, text="ログ")
        frm_log.pack(fill="both", padx=10, pady=5, expand=True)

        self.log_text = tk.Text(frm_log, height=12, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        import sys
        sys.stdout = TextRedirector(self.log_text)

        # ==============================
        # 設定ロード (初期値設定含む)
        # ==============================
        self.load_saved_settings()

        # ==============================
        # 実行ボタンエリア
        # ==============================
        frm_exec = tk.Frame(root)
        frm_exec.pack(pady=10)

        # 処理開始ボタン
        self.btn_run = ttk.Button(
            frm_exec, 
            text="処理開始(既にファイルがあったらスキップ)", 
            command=self.run
        )
        self.btn_run.pack(side="left", padx=10)

        # 処理中断ボタン
        self.btn_stop = ttk.Button(
            frm_exec, 
            text="処理中断", 
            command=self.stop_processing
        )
        self.btn_stop.pack(side="left", padx=10)

    # ---------------------------------------------------------
    def sel_in(self):
        d = filedialog.askdirectory()
        if d:
            self.in_var.set(d)

    def sel_out(self):
        d = filedialog.askdirectory()
        if d:
            self.out_var.set(d)

    # ---------------------------------------------------------
    # class一覧ウィンドウを表示
    # ---------------------------------------------------------
    def open_class_list(self):
        win = tk.Toplevel(self.root)
        win.title("対象物一覧")
        win.geometry("400x600")

        canvas = tk.Canvas(win)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def configure_frame(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", configure_frame)

        tk.Label(scrollable_frame, text="日本語", font=("Arial", 10, "bold"), width=20, borderwidth=1, relief="solid").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(scrollable_frame, text="英語 (これをコピペして)", font=("Arial", 10, "bold"), width=25, borderwidth=1, relief="solid").grid(row=0, column=1, padx=5, pady=5)

        for i, (jp_name, en_name) in enumerate(CLASS_DATA):
            row = i + 1
            lbl_jp = tk.Label(scrollable_frame, text=jp_name, width=20, anchor="w", bg="#f0f0f0")
            lbl_jp.grid(row=row, column=0, padx=5, pady=2, sticky="w")
            
            ent_en = tk.Entry(scrollable_frame, width=25)
            ent_en.insert(0, en_name)
            ent_en.configure(state="readonly") 
            ent_en.grid(row=row, column=1, padx=5, pady=2)
            
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _on_close():
            canvas.unbind_all("<MouseWheel>")
            win.destroy()
            
        win.protocol("WM_DELETE_WINDOW", _on_close)

    # ---------------------------------------------------------
    # TARGET_COLORS を dict 化 (ここにバリデーションを追加)
    # ---------------------------------------------------------
    def get_target_colors(self):
        # バリデーション用にCLASS_DATAの英語名（第2要素）のセットを作成
        valid_class_names = {item[1] for item in CLASS_DATA}

        colors = {}
        for r_idx, row in enumerate(self.cells):
            vals = [e.get().strip() for e in row]
            if vals[0] == "":
                continue
            
            name = vals[0]
            
            # --- 追加したバリデーション処理 ---
            if name not in valid_class_names:
                messagebox.showerror(
                    "エラー", 
                    f"マスク設定 の {r_idx + 1} 行目の 対象物名 '{name}' が不正です。\n"
                    "対象物一覧 に存在する名前を正確に入力してください。"
                )
                return None
            # --------------------------------

            try:
                r = int(vals[1])
                g = int(vals[2])
                b = int(vals[3])
                a = int(vals[4])
                exp = int(vals[5])
                colors[name] = (r, g, b, a, exp)
            except ValueError:
                print(f"無効な行: {vals}")
                messagebox.showerror("エラー", f"マスク設定 の {r_idx + 1} 行目の数値入力が不正です。\n数値を正しく入力してください。")
                return None

        return colors

    # ---------------------------------------------------------
    # 設定保存
    # ---------------------------------------------------------
    def save_current_settings(self):
        data = {
            "input": self.in_var.get(),
            "output": self.out_var.get(),
            "short": self.short_var.get(),
            "long": self.long_var.get(),
            "table": [
                [e.get() for e in row] for row in self.cells
            ],
        }
        save_settings(data)

    # ---------------------------------------------------------
    # 設定ロード (修正: デフォルト値の適用)
    # ---------------------------------------------------------
    def load_saved_settings(self):
        data = load_settings()
        
        if not data:
            print("設定履歴がありません。デフォルト値を適用します。")
            table_data = DEFAULT_TARGETS
        else:
            print("設定履歴を読み込みました")
            self.in_var.set(data.get("input", ""))
            self.out_var.set(data.get("output", ""))
            self.short_var.set(data.get("short", 1024))
            self.long_var.set(data.get("long", 2048))
            table_data = data.get("table", [])

        # テーブルデータの流し込み
        for r, row_vals in enumerate(table_data):
            if r >= len(self.cells):
                break
            for c, val in enumerate(row_vals):
                # 既存のセル内容をクリアしてから挿入
                self.cells[r][c].delete(0, tk.END)
                self.cells[r][c].insert(0, val)

    # ---------------------------------------------------------
    # 中断処理
    # ---------------------------------------------------------
    def stop_processing(self):
        if self.stop_event.is_set():
            return # 既に中断フラグが立っている
        self.stop_event.set()
        print("\n!!! 現在の処理が完了次第停止します !!!\n")

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------
    def run(self):
        # 1. フォルダチェック
        if not self.in_var.get() or not self.out_var.get():
            messagebox.showerror("エラー", "入出力フォルダを指定してください")
            return

        # 2. 解像度入力チェック
        try:
            short_val = self.short_var.get()
            long_val = self.long_var.get()
        except tk.TclError:
            messagebox.showerror("エラー", "推論解像度設定の Shortest または Longest が空欄か、正しくない値です。")
            return

        if long_val <= short_val:
            messagebox.showerror("エラー", "推論解像度設定の Longest は Shortest より大きい値にする必要があります。")
            return

        # 3. 設定保存
        self.save_current_settings()

        # 4. テーブルデータの取得とチェック (ここでclass名のバリデーションも行われる)
        target_colors = self.get_target_colors()
        if target_colors is None:
            return

        # 中断フラグをクリア
        self.stop_event.clear()

        threading.Thread(
            target=start_processing,
            args=(
                self.in_var.get(),
                self.out_var.get(),
                target_colors,
                short_val,
                long_val,
                self.stop_event,  # イベントを渡す
            ),
            daemon=True,
        ).start()


# =========================================================
#  メイン
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    gui = SegGUI(root)
    root.mainloop()
