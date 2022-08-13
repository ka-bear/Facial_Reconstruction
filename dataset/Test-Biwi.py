from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset.biwi import Biwi

if __name__ == '__main__':
    test_ds = Biwi("C:\\Users\\admin\\Downloads\\archive\\faces_0\\")
    test_dl = DataLoader(test_ds, shuffle=True, batch_size=16, num_workers=0)
    print(test_dl)
    for i in test_dl:
        print("hi")
        print(i)
        plt.imshow(i[0][2].numpy().transpose([1, 2, 0]) / 256)
        break
