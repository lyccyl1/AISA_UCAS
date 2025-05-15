import os
import argparse

import numpy as np
import pandas as pd
import ast
from sklearn.metrics import roc_auc_score

def eval_func(gt_name="val_gts", path=""):
    ret = {}

    # 只处理 .xlsx 文件
    files = [f for f in os.listdir(path) if f.endswith(".xlsx")]
    # 去掉 ground-truth 和已有的 LeaderBoard
    files.remove(gt_name + ".xlsx")
    if "LeaderBoard.xlsx" in files:
        files.remove("LeaderBoard.xlsx")

    # 读标签
    gts_df = pd.read_excel(os.path.join(path, gt_name + ".xlsx"))
    gts = gts_df["labels"].values
    gts = np.array(gts)

    for fname in files:
        # 1. 解析 predictions sheet
        pred_df = pd.read_excel(os.path.join(path, fname), sheet_name="predictions")
        raw_preds = pred_df["predictions"].values

        parsed = []
        for p in raw_preds:
            # 如果是 "[...]" 字符串，就转成列表并取第二个元素
            if isinstance(p, str) and p.startswith("["):
                scores = ast.literal_eval(p)
                parsed.append(float(scores[1]))
            else:
                parsed.append(float(p))
        preds = np.array(parsed)

        # 计算 AUC
        auc = roc_auc_score(gts, preds)

        # 2. 读 time sheet，算平均耗时
        time_df = pd.read_excel(os.path.join(path, fname), sheet_name="time")
        mean_time = time_df["Time"].iloc[0] / time_df["Data Volume"].iloc[0]

        team_name = os.path.splitext(fname)[0]
        ret[team_name] = [auc, mean_time]

    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit-path", type=str, default="./res")
    opts = parser.parse_args()

    results = eval_func(path=opts.submit_path)

    # 输出 LeaderBoard.xlsx
    writer = pd.ExcelWriter(os.path.join(opts.submit_path, "LeaderBoard.xlsx"))
    df = pd.DataFrame({
        "Team": list(results.keys()),
        "AUC": [v[0] for v in results.values()],
        "Time": [v[1] for v in results.values()],
    })
    print(df)
    df.to_excel(writer, index=False)
    writer.close()
