import torch
import numpy as np

def train(epoch, 
          model, optimizer, 
          train_loader, 
          criterion_reg, criterion_class, 
          lr_scheduler,
          on_gpu, 
          log_interval, 
          loss_list):
    
    model.train()
    print("lr = ", optimizer.param_groups[0]['lr'])

    if epoch % log_interval == 0:
        running_loss  = torch.tensor(0.0)
        running_loss1 = torch.tensor(0.0)
        running_loss2 = torch.tensor(0.0)
        if on_gpu:
            running_loss, running_loss1, running_loss2  = running_loss.cuda(), running_loss1.cuda(), running_loss2.cuda()

    for batch_idx, current_batch in enumerate(train_loader):
        if on_gpu:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]

        optimizer.zero_grad()
        class_output, regression_output = model(inp)
        regression_gndtruth = current_batch_y[:,0:3]
        class_gndtruth = current_batch_y[:,3].type(torch.LongTensor)
        if on_gpu:       #Seems like reset the tensor type moves data from GPU back to CPU, so need to move it to device again!
            class_gndtruth = class_gndtruth.cuda()

        loss1 = criterion_reg(regression_output, regression_gndtruth)
        loss2 = criterion_class(class_output, class_gndtruth)
        loss  = loss1 + loss2
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            running_loss  += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        lr_scheduler.step()

    if epoch % log_interval == 0:
        running_loss  = running_loss  / len(train_loader)
        running_loss1 = running_loss1 / len(train_loader)
        running_loss2 = running_loss2 / len(train_loader)
        print("epoch: {}, Average loss_reg: {:15.8f}, loss_class: {:15.8f}, loss_tot: {:15.8f}".format(epoch, running_loss1, running_loss2, running_loss))
        loss_list.append(running_loss)

def test(epoch, 
         model,
         test_loader,
         criterion_reg, criterion_class,
         on_gpu,
         log_interval,
         loss_list):

    model.eval()
    
    test_loss  = torch.tensor(0.0)
    test_loss1 = torch.tensor(0.0)
    test_loss2 = torch.tensor(0.0)
    if on_gpu:
        test_loss, test_loss1, test_loss2  = test_loss.cuda(), test_loss1.cuda(), test_loss2.cuda()
    
    for batch_idx, current_batch in enumerate(test_loader):
        if on_gpu:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]        

        with torch.no_grad():
            y_pred_torch_class, y_pred_torch_regression = model(inp)
        regression_gndtruth = current_batch_y[:,0:3]
        class_gndtruth = current_batch_y[:,3].type(torch.LongTensor)
        if on_gpu:       #Seems like reset the tensor type moves data from GPU back to CPU, so need to move it to device again!
            class_gndtruth = class_gndtruth.cuda()

        test_loss1 += criterion_reg(y_pred_torch_regression, regression_gndtruth).item()
        test_loss2 += criterion_class(y_pred_torch_class, class_gndtruth).item()
    
    test_loss1 = test_loss1.detach().cpu().numpy() / len(test_loader)
    test_loss2 = test_loss2.detach().cpu().numpy() / len(test_loader)
    test_loss  = test_loss1 + test_loss2
    if epoch % log_interval == 0:
        print("epoch: {}, Average test_loss_reg: {:15.8f}, test_loss_class: {:15.8f}, test_loss_tot: {:15.8f}".format(epoch, test_loss1, test_loss2, test_loss))
    loss_list.append(test_loss)

    return test_loss

#It will move the model back to CPU!! Be careful!
#Make sure that this is running on CPU, both model and data!!
#Because of that make sure it is running only once! Rather than once every some epoch!
#The inputs are torch tensor, not numpy array or torch dataset
def validation(model, x_test, y_test,
               criterion_class):

    model.eval()
    model = model.cpu()

    with torch.no_grad():
        y_pred_class, y_pred_regression = model(x_test)
    y_pred_class = y_pred_class.detach()
    y_pred_regression = y_pred_regression.detach().reshape(-1, 4)
    avg_diff_0 = torch.mean((y_pred_regression[:,0:3] - y_test[:,0:3]) ** 2).numpy()
    avg_sigma2 = torch.mean(torch.exp(y_pred_regression[:,3])).numpy()
    avg_class_loss = criterion_class(y_pred_class, y_test[:,3].type(torch.LongTensor)).numpy()
    print("Avg diff on test set = ", avg_diff_0)
    print("Avg sigma^2 on test set = ", avg_sigma2)
    print("Avg class loss on test set = ", avg_class_loss)
    
    return avg_diff_0, avg_sigma2, avg_class_loss
