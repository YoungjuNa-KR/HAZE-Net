import utility
import data
import model
import loss
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
from multiprocessing.spawn import freeze_support
from GazeModule.gaze import GazeModel

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    gaze_model = GazeModel(args).cuda()
    loss = loss.Loss(args, checkpoint) 
    # t = Trainer(args, loader, model, loss, checkpoint)
    # t = Trainer(args, loader, gaze_model, loss, checkpoint)
    t = Trainer(args, loader, model, gaze_model, loss, checkpoint)

    def main():
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

    if __name__ == '__main__':  
        freeze_support()  
        main()
