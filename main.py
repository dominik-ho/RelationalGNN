import torch
from torch import Tensor
from gnn import RelationalGNN

def main():
    model = RelationalGNN()
    model = model.cuda()
    x = torch.randint(-10, 10, (10,)).cuda()
    print(x)
    y = model(x)
    print(y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()