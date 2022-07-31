class FlameDataset(Dataset):
    def __init__(self, flame_root, transform=None, test=False):
        self.transform = transform
        self.flame_root = flame_root
        self.test = test

        self.renderlst = os.listdir()
        self.shapelst = os.listdir()

    def __len__(self):
        return len(self.renderlst)

    def __getitem__(self, idx):
        if (self.test):
            filename = f"render_{40 + idx // 8}_{idx % 8}.png"
            filename1 = f"shape_{40 + idx // 8}.npy"

            image = read_image(self.flame_root + "/render//" + filename, ImageReadMode.RGB)
            image = image.type(torch.FloatTensor)
            shape = torch.from_numpy(np.load(self.flame_root + "/shape//" + filename1)[1, :300])
        else:
            filename = f"render_{idx // 8}_{idx % 8}.png"
            filename1 = f"shape_{idx // 8}.npy"

            image = read_image(self.flame_root + "/render//" + filename, ImageReadMode.RGB)
            image = image.type(torch.FloatTensor)
            shape = torch.from_numpy(np.load(self.flame_root + "/shape//" + filename1)[1, :300])
        if self.transform:
            image = self.transform(image)
        return image, shape

