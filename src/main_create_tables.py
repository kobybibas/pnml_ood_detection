import fileinput
import os.path as osp
import sys
from glob import glob

import pandas as pd

sys.path.append("../src")


def replace_string_in_file(file_path: str, src: str, target: str):
    with fileinput.FileInput(file_path, inplace=True) as file:
        for line in file:
            print(line.replace(src, target), end='')



def manipulate_output_file(out_file):
    src = "baseline/+pnml"
    target = "Baseline/+pNML"
    replace_string_in_file(out_file, src, target)

    src = "odin/+pnml"
    target = "ODIN/+pNML"
    replace_string_in_file(out_file, src, target)

    src = "gram/+pnml"
    target = "Gram/+pNML"
    replace_string_in_file(out_file, src, target)

    src = "     &           &"
    target = "IND & OOD &"
    replace_string_in_file(out_file, src, target)

    src = "&      &"
    target = "IND & OOD &"
    replace_string_in_file(out_file, src, target)

    src = "IND & OOD &                       &                       &                       \\\\"
    target = ""
    replace_string_in_file(out_file, src, target)

    src = "\cline{1-5}"
    target = "\midrule"
    replace_string_in_file(out_file, src, target)

    src = "{lllll}"
    target = "{clccc}"
    replace_string_in_file(out_file, src, target)

    src = "%"
    target = "\%"
    replace_string_in_file(out_file, src, target)

    src = "energy/+pnml"
    target = "Energy/+pNML"
    replace_string_in_file(out_file, src, target)

    src = "IND & OOD &          Energy/+pNML &          Energy/+pNML &          Energy/+pNML \\\\"
    target = ""
    replace_string_in_file(out_file, src, target)

    src = '\midrule'
    target =  " \midrule & & \multicolumn{3}{c}{Energy/+pNML} \\\\ \cmidrule{3-5} "
    replace_string_in_file(out_file, src, target)

def prepare_to_latex(df_metric, metric, method, ind_name) -> pd.DataFrame:
    # Change columns order such that pNML is thf first one
    cols = df_metric.columns.tolist()
    cols = cols[::-1]
    df_metric = df_metric[cols]

    # Bold the highest value
    df_metric = df_metric.round(1)
    idx_maxs = df_metric.idxmax(axis=1)
    df_bold = df_metric.applymap(lambda x: '{:.1f}'.format(x))
    df_bold = df_bold.applymap(lambda x: "100" if x == "100.0" else x)
    for loc, col in idx_maxs.items():
        value = df_bold.loc[loc][col]
        df_bold.at[loc, col] = "\textbf{{{}}}".format(value)

    # Change back to columns order
    df_bold = df_bold[cols[::-1]]

    # Prepare latex table
    series = df_bold[f"{metric}_{method}"] + " / " + df_bold[f"{metric}_pnml"]
    df_for_latex = pd.DataFrame(
        {f"{method}/+pnml": series.values},
        index=[[ind_name] * len(series.index), series.index],
    )
    df_for_latex = df_for_latex.rename_axis(("IND", "OOD"))
    df_for_latex = df_for_latex.rename(index={"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "svhn": "SVHN"})
    return df_for_latex


def load_csv_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.replace({"LSUN": "LSUN (C)",
                     "LSUN_resize": "LSUN (R)",
                     "Imagenet_resize": "Imagenet (R)",
                     "Imagenet": "Imagenet (C)",
                     "cifar10": "CIFAR-10",
                     "cifar100": "CIFAR-100",
                     "svhn": "SVHN"})
    df = df.set_index("ood_name")

    return df


def load_latest_results(output_path, method, model_name, ind_name) -> pd:
    csvs = sorted(glob(osp.join(output_path, f"{method}_{model_name}_{ind_name}_*", "performance.csv")))

    # Get most recent
    csv_path = csvs[-1]
    df_csv = load_csv_results(csv_path)
    return df_csv


def create_performance_df(methods, model_name, ind_names, metric, output_path) -> pd.DataFrame:
    dfs_per_method = []
    for method in methods:
        dfs_per_ind = []
        for ind_name in ind_names:
            print(f"{metric} {method} {ind_name}")
            df_csv = load_latest_results(output_path, method, model_name, ind_name)

            # Get desired metric
            df_metric = df_csv[[f"{metric}_{method}", f"{metric}_pnml"]]

            # Bold the best results
            df_for_latex = prepare_to_latex(df_metric, metric, method, ind_name)
            dfs_per_ind.append(df_for_latex)

        df_for_all_inds = pd.concat(dfs_per_ind, axis=0)
        dfs_per_method.append(df_for_all_inds)

    return pd.concat(dfs_per_method, axis=1)


def create_tables():
    methods = ["baseline", "odin", "gram"]
    model_names = ["densenet", "resnet"]
    ind_names = ["cifar10", "cifar100", "svhn"]
    metrics = ["AUROC", "TNR at TPR 95%", "Detection Acc."]

    output_path = "../outputs"

    for metric in metrics:
        for model_name in model_names:
            df = create_performance_df(methods, model_name, ind_names, metric, output_path)
            print(df)

            # Save table
            out_path = osp.join(output_path, f"{metric}_{model_name}.tex")
            df.to_latex(out_path,
                        index=True, na_rep="", multirow=True, escape=False)
            manipulate_output_file(out_path)

    methods = ["energy"]
    model_name = "wrn"
    ind_names = ["cifar10", "cifar100"]
    metrics = ["AUROC", "TNR at TPR 95%", "Detection Acc."]

    df_metrics = []
    for metric in metrics:
        df = create_performance_df(methods, model_name, ind_names, metric, output_path)
        df_metrics.append(df)
    df = pd.concat(df_metrics, axis=1,keys=metrics)

    # Save table
    out_path = osp.join(output_path, f"{model_name}.tex")
    df.to_latex(out_path,
                index=True, na_rep="", multirow=True, escape=False, sparsify=True)
    manipulate_output_file(out_path)


if __name__ == "__main__":
    create_tables()
