import numpy as np
import os
import re
import binvox_rw
from tqdm import tqdm
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from pandas.plotting import table
import time

def get_default_path(model):
    
    if model == 'base' or model == 'orion':
        return 'D:/NNDL/modelnet10/ModelNet10/'
    elif model == 'pointnet':
        return 'D:/NNDL/modelnet10_pointnet/ModelNet10/'
    else:
        raise ValueError('model parmeter must be *base*, *orion* or *pointnet*')

       
        
def get_labels(main_path):
    subdirs = [x for x in os.listdir(main_path)]
    labels = [x for x in subdirs if '.' not in x]
    return labels

def read_off(filename, npoints):
    f = open(filename)
    f.readline()  
    length = f.readline().split(' ')
    points = int(length[0])
    faces = int(length[1])
    All_points = []
    All_faces = []
    for i in range(points):
        new_line = f.readline()
        x = new_line.split(' ')
        A = np.array(x[0:3], dtype='float32')
        All_points.append(A)
    choice = np.random.choice(points, npoints, replace=True)
    all_points = np.array(All_points, dtype=np.float32)
    return all_points[choice, :]

def read_binvox(PATH, orientation=False):
    
    with open(PATH, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    
    data = np.array(model.data*1, dtype=np.float32)
    
    if orientation:
        rotation = re.split(r'[._]+', PATH)[-2]
    
    else:
        rotation = 0
    
    return data.reshape(1, 28, 28, 28), rotation

def get_orientation_label(label, rot):
    
    orientation_classes = {'bed' : 12, 'monitor': 12, 'desk': 12, 'chair': 12, 'dresser': 12, 'toilet': 12, 'sofa': 12, 'table': 3, 'night_stand': 12, 'bathtub': 6 }
    rot = int(rot)
    if ((rot % 30) != 0) or (rot < 0) or (rot > 360):
        raise ValueError('Wrong Rotation')
    rot_label = rot/(360/orientation_classes[label])
    idx = list(orientation_classes).index(label)
    orientation_label = np.sum(list(orientation_classes.values())[:idx]) + rot_label
    
    return int(orientation_label)

    
def train_base(model, optimizer, save_weigths, train_loader, val_loader=None, EPOCHS=10, name='base'):
    inference_time = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    
    for s in range(EPOCHS):
        running_loss = 0.0
        
        model.train()
        iterator = tqdm(train_loader)
        for i, data in enumerate(iterator, 0):
            inputs, labels, rotation = data
            inputs, labels, rotation = inputs.to(device), labels.to(device), rotation.to(device)
            start_time = time.time()
            pred_class = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_time.append(time.time()-start_time)
            loss = loss_fn(pred_class, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iterator.set_description('Epoch: {}, loss: {:.4f}'.format(s+1, running_loss/(i+1)))
           
        #scheduler.step()
        
        model.eval()
        
        if val_loader:
            pred_label = []
            real_labels = []
            iterator = tqdm(val_loader)
            for i, data in enumerate(iterator, 0):
                inputs, labels, rotation = data
                inputs, labels, rotation = inputs.to(device), labels.to(device), rotation.to(device)
                pred_class = model(inputs)
                pred_label.append(pred_class.detach())
                real_labels.append(labels.detach())
                loss = loss_fn(pred_class, labels)
                label_acc = (torch.argmax(torch.softmax(pred_class, dim=1), dim=1).round() == labels).float().mean()
                
                
            pred_label = torch.cat(pred_label, axis=0).to(torch.float32)
            real_labels = torch.cat(real_labels, axis=0).to(torch.float32)
            label_acc = (torch.argmax(torch.softmax(pred_label, dim=1), dim=1).round() == real_labels).float().mean()
            print('Label accuracy: {:.4f}, Loss {:.4f}'.format(label_acc, loss))

        else:
            label_acc = 0
        
                
        # save the model
        file = 'weights/' + name + '_epoch_' + str(s+1)  + ".pth"
        if s % save_weigths == 0:
            torch.save({
                    'epoch': s,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': running_loss,
                    'val_loss': loss,
                    'label_acc': label_acc,
                    }, file)
        else:
            torch.save({
                    'epoch': s,
                    'train_loss': running_loss,
                    'val_loss': loss,
                    'label_acc': label_acc,
                    }, file)
        
    print('Average Inference Time:', np.sum(inference_time)/EPOCHS)
    print('Std of Infernce Time:', np.std(inference_time))
          
def train_orion(model, optimizer, save_weigths, train_loader, val_loader=None, EPOCHS=10, gamma=0.5, name='orion'):
    inference_time = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    
    for s in range(EPOCHS):
        running_loss = 0.0
        
        model.train()
        iterator = tqdm(train_loader)
        for i, data in enumerate(iterator, 0):
            inputs, labels, rotation = data
            inputs, labels, rotation = inputs.to(device), labels.to(device), rotation.to(device)
            start_time = time.time()
            pred_class, pred_orientation = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_time.append(time.time()-start_time)
            loss_class = loss_fn(pred_class, labels)
            loss_orientation = loss_fn(pred_orientation, rotation)
            loss = (1-gamma) * loss_class + gamma * loss_orientation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iterator.set_description('Epoch: {}, loss: {:.4f}'.format(s+1, running_loss/(i+1)))
           
        #scheduler.step()
        
        model.eval()
        
        if val_loader:
            iterator = tqdm(val_loader)
            pred_label = []
            real_labels = []
            pred_rot = []
            real_rot = []
            for i, data in enumerate(iterator, 0):
                inputs, labels, rotation = data
                inputs, labels, rotation = inputs.to(device), labels.to(device), rotation.to(device)
                pred_class, pred_orientation = model(inputs)
                pred_label.append(pred_class.detach())
                real_labels.append(labels.detach())
                pred_rot.append(pred_orientation.detach())
                real_rot.append(rotation.detach())
                loss_class = loss_fn(pred_class, labels)
                loss_orientation = loss_fn(pred_orientation, rotation)
                loss = (1-gamma) * loss_class + gamma * loss_orientation

                    
            pred_label = torch.cat(pred_label, axis=0).to(torch.float32)
            real_labels = torch.cat(real_labels, axis=0).to(torch.float32)
            pred_rot = torch.cat(pred_rot, axis=0).to(torch.float32)
            real_rot = torch.cat(real_rot, axis=0).to(torch.float32)
            label_acc = (torch.argmax(torch.softmax(pred_label, dim=1), dim=1).round() == real_labels).float().mean()
            orient_acc = (torch.argmax(torch.softmax(pred_rot, dim=1), dim=1).round() == real_rot).float().mean()
            print('Label accuracy: {:.4f}, Loss {:.4f}, Orient. acc: {:.4f}'.format(label_acc, loss, orient_acc))
              
        else:
            label_acc = 0
            orient_acc = 0
  
        # save the model
        file = 'weights/' + name + '_epoch_' + str(s+1) + ".pth"
        if s % save_weigths == 0:
            torch.save({
                    'epoch': s,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': running_loss,
                    'val_loss': loss,
                    'orient_acc': orient_acc,
                    'label_acc': label_acc,
                    }, file)
        else:
            torch.save({
                    'epoch': s,
                    'train_loss': running_loss,
                    'val_loss': loss,
                    'orient_acc': orient_acc,
                    'label_acc': label_acc,
                    }, file)
    
    print('Average Inference Time:', np.sum(inference_time)/EPOCHS)
    print('Std of Infernce Time:', np.std(inference_time))
    
def train_pointnet(model, optimizer, save_weigths, train_loader, val_loader=None, EPOCHS=10, name='pointnet'):
    inference_time = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    
    for s in range(EPOCHS):
        running_loss = 0.0
        
        model.train()
        iterator = tqdm(train_loader)
        for i, data in enumerate(iterator, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            pred_class = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_time.append(time.time()-start_time)
            loss = loss_fn(pred_class, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iterator.set_description('Epoch: {}, loss: {:.4f}'.format(s+1, running_loss/(i+1)))
           
        #scheduler.step()
        
        model.eval()
        
        if val_loader:
            pred_label = []
            real_labels = []
            iterator = tqdm(val_loader)
            for i, data in enumerate(iterator, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                pred_class = model(inputs)
                pred_label.append(pred_class.detach())
                real_labels.append(labels.detach())
                loss = loss_fn(pred_class, labels)
            pred_label = torch.cat(pred_label, axis=0).to(torch.float32)
            real_labels = torch.cat(real_labels, axis=0).to(torch.float32)
            label_acc = (torch.argmax(torch.softmax(pred_label, dim=1), dim=1).round() == real_labels).float().mean() 
            print('Label accuracy: {:.4f}, Loss {:.4f}'.format(label_acc, loss))
        
        else:
            label_acc = 0

        # save the model
        file = 'weights/' + name + '_epoch_' + str(s+1)  + ".pth"
        if s % save_weigths == 0:
            torch.save({
                    'epoch': s,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': running_loss,
                    'val_loss': loss,
                    'label_acc': label_acc,
                    }, file)
        else:
            torch.save({
                    'epoch': s,
                    'train_loss': running_loss,
                    'val_loss': loss,
                    'label_acc': label_acc,
                    }, file)
     
    
    print('Average Inference Time:', np.sum(inference_time)/EPOCHS)
    print('Std of Infernce Time:', np.std(inference_time))
          
### Results ###

def acc_per_label(train_load, model, name, main_path, mode=False, classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    iterator = tqdm(train_load)
    false_pred = {f'{i}':[] for i in range(classes)}
    label_data = []
    output_data = []
    for i, data in enumerate(iterator, 0):
        
        if mode == 'orion':
            inputs, label, rotation = data
            inputs, label, rotation = inputs.to(device), label.to(device), rotation.to(device)
            outputs, _ = model(inputs)
            label_data.append(label.detach().cpu().numpy())
            output_data.append(outputs.detach().cpu().numpy())
                
        elif mode == 'base':
            inputs, label, rotation = data
            inputs, label, rotation = inputs.to(device), label.to(device), rotation.to(device)
            outputs = model(inputs)
            label_data.append(label.detach().cpu().numpy())
            output_data.append(outputs.detach().cpu().numpy())
            
        elif mode == 'pointnet':
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)
            outputs = model(inputs)
            label_data.append(label.detach().cpu().numpy())
            output_data.append(outputs.detach().cpu().numpy())
            
        else:
            raise ValueError('Mode must be *orion*, *base* or *pointnet*')
            
            
    label = np.concatenate(label_data, axis=0)
    pred_prob = np.concatenate(output_data, axis=0)
    outputs = np.argmax(pred_prob, axis=1)
    labels = get_labels(main_path)
    fig = plt.figure()
    sns.heatmap(metrics.confusion_matrix(label, outputs), annot = True, xticklabels = labels, yticklabels = labels, cmap = 'flare')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(name)
    plt.savefig(name + '_heatmap', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    
    for i, s in zip(label, outputs):
        false_pred.update({f'{i}':[*false_pred[f'{i}'] ,int(s)]})
    
    for i in false_pred.items():    
        label_, cnts = np.unique(i[1], return_counts=True)
        false_pred[i[0]] = cnts[label_==int(i[0])]/len(i[1])
        
    fig, ax = plt.subplots()
    colors = plt.cm.BuPu(np.linspace(0.3, 0.8, len(false_pred)))
    for row in range(len(false_pred)):
        try:
            p = ax.bar(labels[row], float('%.2f'%false_pred[f'{row}']), 0.7, bottom=0.01, color=colors[row])
        except:
            p = ax.bar(labels[row], 0, 0.7, bottom=0.01, color=colors[row])
        ax.bar_label(p, label_type='center')
    ax.set_xticklabels(labels, fontsize=8, rotation=30)  
    plt.title('Accuracy per Label ' + name)
    plt.ylabel('Accuracy')
    plt.xlabel('Label')
    plt.savefig(name + '_acc_per_label', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    
    return label, outputs, pred_prob
    
def Roc_OvR(y_true, y_pred, y_score, name, main_path):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    label_binarizer = LabelBinarizer().fit(y_true)
    y_onehot_test = label_binarizer.transform(y_pred)
    
    #OvR single label
    labels = get_labels(main_path)
    colors = plt.cm.tab20c(np.linspace(0.1, 0.9, len(labels)))
    for i in range(len(labels)):
        metrics.RocCurveDisplay.from_predictions(
        y_onehot_test[:, i],
        y_score[:, i],
        name=f"{labels[i]}",
        color=colors[i],
        ax=ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_title("One-vs-Rest ROC " + name)
    ax.set_ylabel("True Positive Rate")
    ax.set_ylim([0.2, 1.05])
    ax.set_xlim([-0.03, 0.25])
    plt.legend(prop={'size': 9})
    plt.savefig(name + '_ROC_OvR', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.grid()
    plt.show()
    
    
def ROC_averaged(y_true, y_pred, y_score, name, main_path):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    label_binarizer = LabelBinarizer().fit(y_true)
    y_onehot_test = label_binarizer.transform(y_pred)
    
    #Micro-averaged OvR
    
    metrics.RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    y_score.ravel(),
    name="Micro-averaged OvR " + name,
    color="darkorange",
    ax=ax)
    
    
    #Macro-averaged OvR
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(10):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(10):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= 10

    fpr= fpr_grid
    tpr= mean_tpr
    roc_auc = metrics.auc(fpr, tpr)
    
    ax.plot(fpr, tpr, label=f"Macro-averaged (AUC = {'%.2f'%roc_auc})")
    
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    ax.axis("square")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC-Averaged Micro vs Macro "+name)
    ax.grid()
    fig.savefig(name + '_ROC_averaged', dpi=300, bbox_inches='tight', pad_inches=0.5)
    ax.legend()
    plt.show()
    
    
def metric_scores(y_true, y_pred, y_score, name, main_path):
    
    
    ax = plt.subplot(111, frame_on=False) # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)
    
    label_binarizer = LabelBinarizer().fit(y_true)
    y_onehot_test = label_binarizer.transform(y_pred)
    
    f1_micro, recall_micro, avg_prec_micro = metrics.f1_score(y_true, y_pred, average='micro'), metrics.recall_score(y_true, y_pred, average='micro'), metrics.average_precision_score(y_onehot_test, y_score, average='micro')
    
    f1_macro, recall_macro, avg_prec_macro = metrics.f1_score(y_true, y_pred, average='macro'), metrics.recall_score(y_true, y_pred, average='macro'), metrics.average_precision_score(y_onehot_test, y_score, average='macro')
    
    f1_weighted, recall_weighted, avg_prec_weighted = metrics.f1_score(y_true, y_pred, average='weighted'), metrics.recall_score(y_true, y_pred, average='weighted'), metrics.average_precision_score(y_onehot_test, y_score, average='weighted')
    
    metric = {'Micro':[f1_micro, recall_micro, avg_prec_micro] ,'Macro': [f1_macro, recall_macro, avg_prec_macro], 'Weighted':[f1_weighted, recall_weighted, avg_prec_macro]}
    df = pd.DataFrame.from_dict(metric, orient='index', columns=['F1', 'Recall', 'Avg Precision'])
    result_table = table(ax, df)
    
    plt.savefig(name + '_metrics', dpi=300, bbox_inches='tight', pad_inches=0.5)
    
    return result_table
    
def make_results(model, val_load, name, main_path, mode=False, classes=10):
    y_true, y_pred, y_score = acc_per_label(val_load, model, name, main_path, mode=mode, classes=classes)
    Roc_OvR(y_true, y_pred, y_score, name, main_path)
    ROC_averaged(y_true, y_pred, y_score, name, main_path)
    metric_scores(y_true, y_pred, y_score, name, main_path)
 