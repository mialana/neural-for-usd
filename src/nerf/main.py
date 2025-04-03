from helpers import init_models
from trainer import train
from imports import n_restarts, warmup_min_fitness

from imports import torch, signal, sys

if __name__ == "__main__":
    model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
    
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        torch.save(model.state_dict(), "data/nerf.pt")
        torch.save(fine_model.state_dict(), "data/nerf-fine.pt")
        torch.save(optimizer.state_dict(), "data/optimizer.pt")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for i in range(n_restarts):
        model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
        
        success, train_psnrs, val_psnrs = train(model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper)
        if success and val_psnrs[-1] >= warmup_min_fitness:
            print("Training successful!")
            break
        else:
            print(f"Restarting... this is attempt {i}")

    print("")
    print(f"Done!")

    torch.save(model.state_dict(), "data/nerf.pt")
    torch.save(fine_model.state_dict(), "data/nerf-fine.pt")
