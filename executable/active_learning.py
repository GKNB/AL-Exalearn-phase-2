import torch
import numpy as np
import argparse

import model as mdp

def get_freq(args, do_print = True):
    checkpoint = torch.load('ckpt.pth')
    model = mdp.FullModel(len_input = 2806, num_hidden = 256, num_output = 3+1, num_classes = 3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.cpu()

    x_study_torch = torch.load("x_study_torch.pt")
    with torch.no_grad():
        y_pred_torch_class, y_pred_torch_regression = model(x_study_torch)
    
    y_pred_np = y_pred_torch_regression.detach().numpy().reshape(-1, 4)
    w_reg = np.exp(y_pred_np[:,3])
    w_reg = w_reg.astype(np.float64)
    w_reg = w_reg / np.sum(w_reg)
    prob = torch.nn.functional.softmax(y_pred_torch_class, dim=1)
    entropy = -torch.sum(prob * torch.log(prob), dim=1).detach().numpy()
    w_class = entropy / np.sum(entropy)

    w = 0.5 * w_reg + 0.5 * w_class
    freq = np.random.multinomial(args.num_new_sample, w)
    
    if do_print:
        with np.printoptions(threshold=np.inf):
            print("logits = ", y_pred_torch_class.numpy())
            print("prob = ", prob.numpy())
            print("entropy = ", entropy)
            print("logsig2 = ", y_pred_np[:,3])
            print("sig2 after norm = ", w_reg)
            print("entropy after norm = ", w_class)
            print("freq = ", freq)
            print("freq.shape = ", freq.shape)
            print("freq.sum = ", np.sum(freq))
    
    np.save('AL-freq.npy', freq)

def main():
    parser = argparse.ArgumentParser(description='Exalearn_AL_v1')
    
    parser.add_argument('--seed',           type=int,   required=True,
                        help='random seed (default: 42)')
    parser.add_argument('--num_new_sample',         type=int,   required=True,
                        help='number of new samples for next simulation (default: 2000)')
    parser.add_argument('--policy',         choices=['uncertainty', 'loss', 'random'],
                        help='AL policy used. uncertainty is the one we want to look at, random means randomly sample')

    args = parser.parse_args()

    get_freq(args, True)

if __name__ == "__main__":
    main()
