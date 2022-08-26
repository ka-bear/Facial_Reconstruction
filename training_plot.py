import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    state = torch.load(r"C:\Users\admin\PycharmProjects\Facial_Reconstruction1\modelbiwi.pt")
    train = state["metrics"]["train"]
    val = state["metrics"]["val"]
    train = np.array([i.cpu().numpy() for i in train]) / 628
    val = np.array([i.cpu().numpy() for i in val]) / 157
    data = pd.DataFrame({
        "epoch": np.concatenate([np.arange(1, 31), np.arange(1, 31)]),
        "log_loss": np.log(np.concatenate([train, val])),
        "split": ["train"] * 30 + ["val"] * 30
    })
    sns.lineplot(x="epoch", y="log_loss", hue="split", data=data)
    plt.show()

if __name__ == "__main__":
    main()
