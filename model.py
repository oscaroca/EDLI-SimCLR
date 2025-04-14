import os
import torch

from simclr import SimCLR
from simclr.modules import LARS


def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(args, model, optimizer):
    # Define the output paths for both .tar and .pth files
    out_tar = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))
    out_pth = os.path.join(args.model_path, "resnet_model_{}.pth".format(args.current_epoch))

    # Save as .tar
    if isinstance(model, torch.nn.DataParallel):
        torch.save({'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': args.current_epoch}, out_tar)
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': args.current_epoch}, out_tar)

    # Save as .pth (only the model's state_dict)
    if isinstance(model, torch.nn.DataParallel):
        state = {
    
        'model':  model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.current_epoch,
    }
        torch.save(state, out_pth)
        del state
    else:
        state = {
      
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.current_epoch,
        }   
        torch.save(state, out_pth)
        del state
        

