import torch
from src.argument import parse_args


def main():
    args = parse_args() 
    # torch.set_num_threads(4)

    if args.model == "SASRec":
        from models.SASRec import Trainer
        embedder = Trainer(args)

    elif args.model == "HPSERec":
        from models.HPSERec import Trainer
        embedder = Trainer(args)

    
    if args.inference:
        embedder.test()
    else:
        embedder.train()

if __name__ == "__main__":
    main()