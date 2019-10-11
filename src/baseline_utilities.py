import pandas as pd

# ------ #
# ODIN
# ------ #
datasets_odin = ['Imagenet (crop)', 'Imagenet (resize)', 'LSUN (crop)',
                 'LSUN (resize)', 'iSUN', 'Uniform', 'Gaussian']
densnet_cifar10_odin = {

    'Model': ['DensNet-BC CIFAR10'] * len(datasets_odin),
    'OOD Datasets': datasets_odin,
    'Method': ['ODIN'] * len(datasets_odin),
    'FPR (95% TPR) ↓': [4.3, 7.5, 8.7, 3.8, 6.3, 0.0, 0.0],
    'Detection Error ↓': [4.7, 6.3, 6.9, 4.4, 5.7, 2.5, 2.5],
    'AUROC ↑': [99.1, 98.5, 98.2, 99.2, 98.8, 99.9, 100.0],
    'AP-In ↑': [99.1, 98.6, 98.5, 99.3, 98.9, 100.0, 100.0],
    'AP-Out ↑': [99.1, 98.5, 97.8, 99.2, 98.8, 99.9, 100.0],
}
odin_df_densnet_cifar10 = pd.DataFrame(densnet_cifar10_odin)

densnet_cifar100_odin = {
    'Model': ['DensNet-BC CIFAR100'] * len(datasets_odin),
    'OOD Datasets': datasets_odin,
    'Method': ['ODIN'] * len(datasets_odin),
    'FPR (95% TPR) ↓': [17.3, 44.3, 17.6, 44.0, 49.5, 0.5, 0.2],
    'Detection Error ↓': [11.2, 24.6, 11.3, 24.5, 27.2, 2.8, 2.6],
    'AUROC ↑': [97.1, 90.7, 96.8, 91.5, 90.1, 99.5, 99.6],
    'AP-In ↑': [97.4, 91.4, 97.1, 92.4, 91.1, 99.6, 99.7],
    'AP-Out ↑': [96.8, 90.1, 96.5, 90.6, 88.9, 99.0, 99.1]}
odin_df_densnet_cifar100 = pd.DataFrame(densnet_cifar100_odin)

resnet_cifar10_odin = {
    'Model': ['WRN-28-10 CIFAR10'] * len(datasets_odin),
    'OOD Datasets': datasets_odin,
    'Method': ['ODIN'] * len(datasets_odin),
    'FPR (95% TPR) ↓': [23.4, 25.5, 21.8, 17.6, 21.3, 0.0, 0.0],
    'Detection Error ↓': [14.2, 15.2, 13.4, 11.3, 13.2, 2.5, 2.5],
    'AUROC ↑': [94.2, 92.1, 95.9, 95.4, 93.7, 100.0, 100.0],
    'AP-In ↑': [92.8, 89.0, 95.8, 93.8, 91.2, 100.0, 100.0],
    'AP-Out ↑': [94.7, 93.6, 95.5, 96.1, 94.9, 100.0, 100.0]}
odin_df_resnet_cifar10 = pd.DataFrame(resnet_cifar10_odin)

resnet_cifar100_odin = {
    'Model': ['WRN-28-10 CIFAR100'] * len(datasets_odin),
    'OOD Datasets': datasets_odin,
    'Method': ['ODIN'] * len(datasets_odin),
    'FPR (95% TPR) ↓': [43.9, 55.9, 39.6, 56.5, 57.3, 0.1, 1.0],
    'Detection Error ↓': [24.4, 30.4, 22.3, 30.8, 31.1, 2.5, 3.0],
    'AUROC ↑': [90.8, 84.0, 92.0, 86.0, 85.6, 99.1, 98.5],
    'AP-In ↑': [91.4, 82.8, 92.4, 86.2, 85.9, 99.4, 99.1],
    'AP-Out ↑': [90.0, 84.4, 91.6, 84.9, 84.8, 97.5, 95.9]}
odin_df_resnet_cifar100 = pd.DataFrame(resnet_cifar100_odin)

# ------------- #
# Leave one out
# ------------- #
datasets_loo = ['Imagenet (crop)', 'Imagenet (resize)', 'LSUN (crop)', 'LSUN (resize)', 'Uniform', 'Gaussian']
densnet_cifar10_lvo = {
    'Model': ['DensNet-BC CIFAR10'] * len(datasets_loo),
    'OOD Datasets': datasets_loo,
    'Method': ['LOO'] * len(datasets_loo),
    'FPR (95% TPR) ↓': [1.23, 2.93, 3.42, 0.77, 2.61, 0.00],
    'Detection Error ↓': [2.63, 3.84, 4.12, 2.1, 3.6, 0.2],
    'AUROC ↑': [99.65, 99.34, 99.25, 99.75, 98.55, 99.84],
    'AP-In ↑': [99.68, 99.37, 99.29, 99.77, 98.94, 99.86],
    'AP-Out ↑': [99.64, 99.32, 99.24, 99.73, 97.52, 99.6]
}
lvo_df_densnet_cifar10 = pd.DataFrame(densnet_cifar10_lvo)

densnet_cifar100_lvo = {
    'Model': ['DensNet-BC CIFAR100'] * len(datasets_loo),
    'OOD Datasets': datasets_loo,
    'Method': ['LOO'] * len(datasets_loo),
    'FPR (95% TPR) ↓': [8.29, 20.52, 14.69, 16.23, 79.73, 38.52],
    'Detection Error ↓': [6.27, 9.98, 8.46, 8.77, 9.46, 8.21],
    'AUROC ↑': [98.43, 96.27, 97.37, 97.03, 92.0, 94.89],
    'AP-In ↑': [98.58, 96.66, 97.62, 97.37, 94.77, 96.36],
    'AP-Out ↑': [98.3, 95.82, 97.18, 96.6, 83.81, 90.01]}
lvo_df_densnet_cifar100 = pd.DataFrame(densnet_cifar100_lvo)

resnet_cifar10_lvo = {
    'Model': ['WRN-28-10 CIFAR10'] * len(datasets_loo),
    'OOD Datasets': datasets_loo,
    'Method': ['LOO'] * len(datasets_loo),
    'FPR (95% TPR) ↓': [0.82, 2.94, 1.93, 0.88, 16.39, 0.0],
    'Detection Error ↓': [2.24, 3.83, 3.24, 2.52, 5.39, 1.03],
    'AUROC ↑': [99.75, 99.36, 99.55, 99.7, 96.77, 99.58],
    'AP-In ↑': [99.77, 99.4, 99.57, 99.72, 97.78, 99.71],
    'AP-Out ↑': [99.75, 99.36, 99.55, 99.68, 94.18, 99.2]}
lvo_df_resnet_cifar10 = pd.DataFrame(resnet_cifar10_lvo)

resnet_cifar100_lvo = {
    'Model': ['WRN-28-10 CIFAR100'] * len(datasets_loo),
    'OOD Datasets': datasets_loo,
    'Method': ['LOO'] * len(datasets_loo),
    'FPR (95% TPR) ↓': [9.17, 24.53, 14.22, 16.53, 99.9, 98.26],
    'Detection Error ↓': [6.67, 11.64, 8.2, 9.14, 14.86, 16.88],
    'AUROC ↑': [98.22, 95.18, 97.38, 96.77, 83.44, 93.04],
    'AP-In ↑': [98.39, 95.5, 97.62, 97.03, 89.43, 88.64],
    'AP-Out ↑': [98.07, 94.78, 97.16, 96.41, 71.2, 71.62]}
lvo_df_resnet_cifar100 = pd.DataFrame(resnet_cifar100_lvo)
