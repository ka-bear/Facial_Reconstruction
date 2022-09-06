import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    state = torch.load("models/modelbiwi.pt")
    train = state["metrics"]["train"]
    val = state["metrics"]["val"]
    train = np.array([i.cpu().numpy() for i in train]) / 628
    val = np.array([i.cpu().numpy() for i in val]) / 157
    print(train.size, val.size)
    data = pd.DataFrame({
        "epoch": np.concatenate([np.arange(1, 123), np.arange(1, 123)]),
        "loss": np.concatenate([train, val]),
        "split": ["train"] * 122 + ["val"] * 122
    })
    sns.lineplot(x="epoch", y="loss", hue="split", data=data)
    plt.ylim(0, 10)
    plt.show()

if __name__ == "__main__":
    main()
