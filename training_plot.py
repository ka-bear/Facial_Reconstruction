import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    state = torch.load("mobilenet_biwi.pt")
    train = state["metrics"]["train"]
    val = state["metrics"]["val"]
    train = np.log10(np.array([i.cpu().numpy() for i in train]) / 196)
    val = np.log10(np.array([i.cpu().numpy() for i in val]) / 49)
    print(train, val)
    print(train.size, val.size)
    data = pd.DataFrame({
        "epoch": np.concatenate([np.arange(1, train.size + 1), np.arange(1, val.size + 1)]),
        "log(loss)": np.concatenate([train, val]),
        "split": ["train"] * train.size + ["val"] * val.size
    })
    sns.lineplot(x="epoch", y="log(loss)", hue="split", data=data)
    plt.show()


if __name__ == "__main__":
    main()
