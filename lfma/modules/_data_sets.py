from torch.utils.data import Dataset


class MultiAnnotatorDataSet(Dataset):
    """MultiAnnotatorDataSet

    Dataset to deal with samples annotated by multiple annotators.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, *)
        Samples' features whose shape depends on the concrete learning problem.
    y : torch.Tensor of shape (n_samples, *)
        Samples' targets whose shape depends on the concrete learning problem.
    A: torch.Tensor of shape (n_annotators, *)
        Annotators' features whose shape depends on the concrete learning problem.
    """

    def __init__(self, X, y=None, A=None, transform=None):
        super().__init__()
        self.X = X
        self.y = y
        self.A = A
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.transform(self.X[idx]) if self.transform else self.X[idx]
        if self.y is not None and self.A is not None:
            return x, self.y[idx], self.A
        elif self.y is None and self.A is not None:
            return x, self.A
        elif self.y is not None and self.A is None:
            return x, self.y[idx]
        else:
            return x