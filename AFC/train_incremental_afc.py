import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from AFC.dataloader_afc import IceLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from datetime import datetime
import random
from model.layers import NormedLinear, SplitNormedLinear, LSCLinear, SplitLSCLinear

from sklearn.cluster import KMeans

import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def nca(similarities, targets, scale=1, margin=0.6):
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    
    similarities = scale * (similarities - margin)

    similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

    disable_pos = torch.zeros_like(similarities)
    disable_pos[torch.arange(len(similarities)),
                targets] = similarities[torch.arange(len(similarities)), targets]

    numerator = similarities[torch.arange(similarities.shape[0]), targets]
    denominator = similarities - disable_pos

    losses = numerator - torch.log(torch.exp(denominator).sum(-1))
    
    losses = -losses
    loss = torch.mean(losses)
    
    return loss


def imprint_weights(args, model, step, train_loader):
    model.eval()
    
    class_per_step = args.class_num_per_step
    start = step * class_per_step

    features_arr = []
    targets_arr = []
    
    with torch.no_grad():
        print('===========================')
        print('Imprint weights')
        print('===========================')
        model = model.to(device)

        for samples in tqdm(train_loader):
            data, labels = samples
            # labels = labels.to(device)
            if args.modality == 'visual':
                visual = data
                visual = visual.to(device)
                features = model(visual=visual, out_logits=False, out_features=True, out_features_norm=True)
            elif args.modality == 'audio':
                audio = data
                audio = audio.to(device)
                features = model(audio=audio, out_logits=False, out_features=True, out_features_norm=True)
            else:
                visual = data[0]
                audio = data[1]
                visual = visual.to(device)
                audio = audio.to(device)
                features = model(visual=visual, audio=audio, out_logits=False, out_features=True, out_features_norm=True)
            
            features = features.detach().cpu()
            
            new_idx = labels >= start

            new_features = features[new_idx]
            new_target_idx = labels[new_idx] - start
            
            features_arr.append(new_features)
            targets_arr.append(new_target_idx)
        
        model = model.to('cpu')
        weights_norm = model.classifier.fc1.weight.data.norm(dim=1, keepdim=True)
        avg_weights_norm = torch.mean(weights_norm, dim=0)

        features = torch.cat(features_arr, dim=0)
        targets = torch.cat(targets_arr, dim=0)
        
        new_weights = []
        for c in range(class_per_step):
            class_features = features[targets == c]
            clusterizer = KMeans(n_clusters=10)
            clusterizer.fit(class_features.numpy())
            
            for center in clusterizer.cluster_centers_:
                new_weights.append(torch.tensor(center) * avg_weights_norm)
        
        new_weights = torch.stack(new_weights)
        model.classifier.fc2.weight.data = new_weights
    
    return model



def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups: 
            param_group['lr'] = new_lr


def gen_step_exemplar_set(args, model, step, train_data_set, m_per_class, old_exemplar_class_vids):
    model.eval()

    train_data_set._switch_gen_exemplar(old_exemplar_class_vids)
    gen_exemplar_loader = DataLoader(train_data_set, batch_size=args.gen_exem_batch_size, num_workers=args.num_workers,
                                     pin_memory=True, drop_last=False, shuffle=False)

    all_features = torch.Tensor([])
    # all_vids = torch.Tensor([])
    all_vids = []
    all_category_id = torch.Tensor([])
    with torch.no_grad():
        for data, vids, category_id in tqdm(gen_exemplar_loader):
            if args.modality == 'visual':
                visual = data
                visual = visual.to(device)
                features = model(visual=visual, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
            elif args.modality == 'audio':
                audio = data
                audio = audio.to(device)
                features = model(audio=audio, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
            else:
                visual = data[0]
                audio = data[1]
                visual = visual.to(device)
                audio = audio.to(device)
                features = model(visual=visual, audio=audio, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
            all_features = torch.cat((all_features, features), dim=0)
            # print(vids)
            all_vids += vids
            # all_vids = torch.cat((all_vids, vids), dim=0)
            all_category_id = torch.cat((all_category_id, category_id), dim=0)
    
    all_features = all_features.numpy()
    all_vids = np.array(all_vids)
    all_category_id = all_category_id.numpy()

    current_step_class_idxs = np.array(range(args.class_num_per_step * step, args.class_num_per_step * (step + 1)))

    step_all_class_exemplar_features = []
    step_all_class_exemplar_vids = []

    for class_id in current_step_class_idxs:
        class_features = all_features[all_category_id == class_id]
        class_vids = all_vids[all_category_id == class_id]
        class_mean_feature = np.mean(class_features, axis=0)

        class_exemplar_features = []
        class_exemplar_vids = []
        now_class_mean = np.zeros((1, class_features.shape[-1]))
        for i in range(m_per_class):
            if i >= len(class_features):
                class_exemplar_vids.append(None)
                class_exemplar_features.append(np.zeros((class_features.shape[-1])))
                continue
            x = class_mean_feature - (now_class_mean + class_features) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += class_features[index]
            class_exemplar_features.append(class_features[index])
            class_exemplar_vids.append(class_vids[index])

        step_all_class_exemplar_features.append(class_exemplar_features)
        step_all_class_exemplar_vids.append(class_exemplar_vids)
    
    step_all_class_exemplar_features = np.array(step_all_class_exemplar_features)
    step_all_class_exemplar_vids = np.array(step_all_class_exemplar_vids)

    if step == 0:
        exemplar_class_features = step_all_class_exemplar_features
        new_exemplar_class_vids = step_all_class_exemplar_vids
    else:
        old_exemplar_class_vids_ = old_exemplar_class_vids[..., :m_per_class]
        
        exemplar_class_features = []
        for old_class_id in range(len(old_exemplar_class_vids_)):
            old_class_vids = old_exemplar_class_vids_[old_class_id]
            old_class_features = np.array([all_features[all_vids==vid][0] for vid in old_class_vids])
            exemplar_class_features.append(old_class_features)
        exemplar_class_features = np.array(exemplar_class_features)
        
        exemplar_class_features = np.concatenate((exemplar_class_features, step_all_class_exemplar_features), axis=0)
        new_exemplar_class_vids = np.concatenate((old_exemplar_class_vids_, step_all_class_exemplar_vids), axis=0)

    train_data_set._switch_train()

    if step != 0:
        train_data_set._conbine_exemplar()

    return exemplar_class_features, new_exemplar_class_vids


def train(args, step, train_data_set, val_data_set, exemplar_class_vids):
    train_loader = DataLoader(train_data_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)
    
    step_out_class_num = (step + 1) * args.class_num_per_step
    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num, LSC=True)
    elif step == 1:
        model = torch.load('./save/{}/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, args.classify, step-1, args.modality))
        in_features = model.classifier.in_features
        out_features = model.classifier.out_features

        new_classifier = SplitLSCLinear(in_features, out_features, args.class_num_per_step)
        new_classifier.fc1.weight.data = model.classifier.weight.data
        model.classifier = new_classifier

        old_model = copy.deepcopy(model)
    else:
        model = torch.load('./save/{}/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, args.classify, step-1, args.modality))
        in_features = model.classifier.in_features

        out_features1 = model.classifier.fc1.out_features
        out_features2 = model.classifier.fc2.out_features
        K = model.classifier.fc2.K

        new_fc = SplitLSCLinear(in_features, out_features1 + out_features2, args.class_num_per_step)

        new_fc.fc1.weight.data[:out_features1*K] = model.classifier.fc1.weight.data
        new_fc.fc1.weight.data[out_features1*K:] = model.classifier.fc2.weight.data
        model.classifier = new_fc

        old_model = copy.deepcopy(model)
    
    if step > 0:
        model = imprint_weights(args, model, step, train_loader)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step != 0:
            old_model = nn.DataParallel(old_model)
    
    model = model.to(device)
    if step != 0:
        old_model = old_model.to(device)
        old_model.eval()
    
    if step > 0:
        ignored_params = list(map(id, model.classifier.fc1.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        params =[{'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}, \
                 {'params': model.classifier.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
    else:
        params = model.parameters()
    
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0
    best_exemplar_class_vids = None
    best_exemplar_class_mean = None

    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()
        iterator = tqdm(train_loader)
        
        for samples in iterator:
            data, labels = samples
            labels = labels.to(device)
            if args.modality == 'visual':
                visual = data
                visual = visual.to(device)
                out, feature = model(visual=visual, AFC_train_out=True)
                if step != 0:
                    old_out, old_feature = old_model(visual=visual, AFC_train_out=True)
            elif args.modality == 'audio':
                audio = data
                audio = audio.to(device)
                out, feature = model(audio=audio, AFC_train_out=True)
                if step != 0:
                    old_out, old_feature = old_model(audio=audio, AFC_train_out=True)
            else:
                visual = data[0]
                audio = data[1]
                visual = visual.to(device)
                audio = audio.to(device)
                out, visual_pooled_feature, audio_feature, visual_feature = model(visual=visual, audio=audio, AFC_train_out=True)
                
                if step != 0:
                    old_out, old_visual_pooled_feature, old_audio_feature, old_visual_feature = old_model(visual=visual, audio=audio, AFC_train_out=True)
                
            if step != 0:
                prev_loss_cls = nca(old_out, labels, scale=old_model.classifier.factor)
                prev_loss_cls.backward(retain_graph=True)
                if args.modality == 'audio-visual':
                    old_audio_feature_grad = old_audio_feature.grad
                    old_visual_feature_grad = old_visual_feature.grad
                    I_audio = torch.norm(old_audio_feature_grad, p='fro', dim=-1)
                    I_visual = torch.norm(old_visual_feature_grad, p='fro', dim=-1)

                    old_visual_pooled_feature_grad = old_visual_pooled_feature.grad
                    I_visual_pooled = torch.norm(old_visual_pooled_feature_grad, p='fro', dim=-1)
                    all_I = I_audio + I_visual + I_visual_pooled
                    norm_I_visual_pooled = I_visual_pooled / all_I
                    loss_I_visual_pooled = torch.mean(norm_I_visual_pooled * torch.norm(F.normalize(visual_pooled_feature) - F.normalize(old_visual_pooled_feature.detach()), p='fro', dim=-1))

                    norm_I_audio = I_audio / all_I
                    norm_I_visual = I_visual / all_I
                    loss_I_audio = torch.mean(norm_I_audio * torch.norm(F.normalize(audio_feature) - F.normalize(old_audio_feature.detach()), p='fro', dim=-1))
                    loss_I_visual = torch.mean(norm_I_visual * torch.norm(F.normalize(visual_feature) - F.normalize(old_visual_feature.detach()), p='fro', dim=-1))

                    loss_disc = loss_I_visual_pooled + loss_I_audio + loss_I_visual
                else:
                    loss_disc = torch.mean(torch.norm(F.normalize(feature) - F.normalize(old_feature), p='fro', dim=-1))
                loss_cls = nca(out, labels, scale=model.classifier.factor)
                lam_t = (step + 1) ** 0.5
                lam_disc = args.lam_disc
                loss = loss_cls + lam_disc * lam_t * loss_disc
            else:
                loss = nca(out, labels, scale=model.classifier.factor)
            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)

        if args.classify == 'NME':
            m = args.memory_size // (args.class_num_per_step * (step + 1))
            exemplar_class_features, new_exemplar_class_vids = gen_step_exemplar_set(args, model, step, train_data_set, m, exemplar_class_vids)
            exemplar_class_mean = np.mean(exemplar_class_features, axis=1)

        all_val_out = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                if args.modality == 'visual':
                    val_visual = val_data
                    val_visual = val_visual.to(device)
                    if args.classify == 'NME':
                        val_out = model(visual=val_visual, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
                    else:
                        val_out = model(visual=val_visual).detach().cpu()
                elif args.modality == 'audio':
                    val_audio = val_data
                    val_audio = val_audio.to(device)
                    if args.classify == 'NME':
                        val_out = model(audio=val_audio, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
                    else:
                        val_out = model(audio=val_audio).detach().cpu()
                else:
                    val_visual = val_data[0]
                    val_audio = val_data[1]
                    val_visual = val_visual.to(device)
                    val_audio = val_audio.to(device)
                    if args.classify == 'NME':
                        val_out = model(visual=val_visual, audio=val_audio, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
                    else:
                        val_out = model(visual=val_visual, audio=val_audio).detach().cpu()
                        val_out = F.softmax(val_out, dim=-1).detach().cpu()
                all_val_out = torch.cat((all_val_out, val_out), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)
        if args.classify == 'NME':
            all_val_pre = NME_classify(all_val_out, exemplar_class_mean)
            all_val_labels = all_val_labels.numpy()
            val_res = (all_val_pre == all_val_labels).sum() / len(all_val_labels)
        else:
            val_res = top_1_acc(all_val_out, all_val_labels)
        val_acc_list.append(val_res)
        print('Epoch:{} val_res:{:.6f} '.format(epoch, val_res), flush=True)

        if val_res > best_val_res:
            best_val_res = val_res

            if args.classify == 'NME':
                best_exemplar_class_vids = new_exemplar_class_vids
                best_exemplar_class_mean = exemplar_class_mean

            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            if torch.cuda.device_count() > 1: 
                torch.save(model.module, './save/{}/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, args.classify, step, args.modality))
            else:
                torch.save(model, './save/{}/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, args.classify, step, args.modality))
        
        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/{}/{}/train_loss_step_{}.png'.format(args.dataset, args.modality, args.classify, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/{}/{}/val_acc_step_{}.png'.format(args.dataset, args.modality, args.classify, step))
        plt.close()

        if args.lr_decay:
            adjust_learning_rate(args, opt, epoch)
        
    if args.classify == 'LSC':
        step_best_model = torch.load('./save/{}/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, args.classify, step, args.modality))
        m = args.memory_size // (args.class_num_per_step * (step + 1))
        best_exemplar_class_features, best_exemplar_class_vids = gen_step_exemplar_set(args, step_best_model, step, train_data_set, m, exemplar_class_vids)
        best_exemplar_class_mean = np.mean(best_exemplar_class_features, axis=1)
    np.save('./save/{}/{}/{}/step_{}_best_{}_exemplar_class_mean.npy'.format(args.dataset, args.modality, args.classify, step, args.modality), best_exemplar_class_mean)
    
    return best_exemplar_class_vids, best_exemplar_class_mean
    
def NME_classify(features, exemplar_class_mean):
    norm_features = F.normalize(features).numpy()
    all_res = []
    for f in norm_features:
        dis = f - exemplar_class_mean
        dis = np.linalg.norm(dis, ord=2, axis=1)
        pre = np.argmin(dis)
        all_res.append(pre)
    return np.array(all_res)

def detailed_test(args, step, test_data_set, task_best_acc_list):
    print("=====================================")
    print("Start testing...")
    print("=====================================")
    # model = IncreAudioVisualNet(args=args)

    model = torch.load('./save/{}/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, args.classify, step, args.modality))
    
    exemplar_class_mean = np.load('./save/{}/{}/{}/step_{}_best_{}_exemplar_class_mean.npy'.format(args.dataset, args.modality, args.classify, step, args.modality))

    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    
    all_test_out = torch.Tensor([])
    all_test_labels = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            # test_labels = test_labels.to(device)
            if args.modality == 'visual':
                test_visual = test_data
                test_visual = test_visual.to(device)
                if args.classify == 'NME':
                    test_out = model(visual=test_visual, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
                else:
                    test_out = model(visual=test_visual).detach().cpu()
            elif args.modality == 'audio':
                test_audio = test_data
                test_audio = test_audio.to(device)
                if args.classify == 'NME':
                    test_out = model(audio=test_audio, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
                else:
                    test_out = model(audio=test_audio).detach().cpu()
            else:
                test_visual = test_data[0]
                test_audio = test_data[1]
                test_visual = test_visual.to(device)
                test_audio = test_audio.to(device)
                if args.classify == 'NME':
                    test_out = model(visual=test_visual, audio=test_audio, out_logits=False, out_features=True, out_features_norm=True).detach().cpu()
                else:
                    test_out = model(visual=test_visual, audio=test_audio).detach().cpu()
                    test_out = F.softmax(test_out, dim=-1).detach().cpu()
            # test_out = F.softmax(test_out, dim=-1).detach().cpu()
            all_test_out = torch.cat((all_test_out, test_out), dim=0)
            all_test_labels = torch.cat((all_test_labels, test_labels), dim=0)
    
    if args.classify == 'NME':
        all_test_pre = NME_classify(all_test_out, exemplar_class_mean)
        # all_test_labels = all_test_labels.numpy()
        test_res = (all_test_pre == all_test_labels.numpy()).sum() / len(all_test_labels)

    else:
        test_res = top_1_acc(all_test_out, all_test_labels)
    
    print("Incremental step {} Testing res: {:.6f}".format(step, test_res))
    
    old_task_acc_list = []
    for i in range(step+1):
        step_class_list = range(i*args.class_num_per_step, (i+1)*args.class_num_per_step)
        step_class_idxs = []
        for c in step_class_list:
            idxs = np.where(all_test_labels.numpy() == c)[0].tolist()
            step_class_idxs += idxs
        step_class_idxs = np.array(step_class_idxs)
        i_labels = torch.Tensor(all_test_labels.numpy()[step_class_idxs])
        if args.classify == 'NME':
            i_pred = all_test_pre[step_class_idxs]
            i_acc = (i_pred == i_labels.numpy()).sum() / len(i_labels)
        else:
            i_outs = torch.Tensor(all_test_out.numpy()[step_class_idxs])
            i_acc = top_1_acc(i_outs, i_labels)

        if i == step:
            curren_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)
    if step > 0:
        forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        print('forgetting: {:.6f}'.format(forgetting))
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None
    task_best_acc_list.append(curren_step_acc)

    return forgetting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AVE', choices=['AVE', 'ksounds', 'VGGSound_100'])
    parser.add_argument('--modality', type=str, default='visual', choices=['visual', 'audio', 'audio-visual'])
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=32)

    parser.add_argument('--gen_exem_batch_size', type=int, default=64)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[500], nargs='+', help="")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--class_num_per_step', type=int, default=7)

    parser.add_argument('--memory_size', type=int, default=340)

    parser.add_argument('--classify', type=str, default='NME', choices=['LSC', 'NME'])
    parser.add_argument('--lam_disc', type=float, default=4.0)

    args = parser.parse_args()
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step

    setup_seed(args.seed)
    
    print('Training start time: {}'.format(datetime.now()))

    train_set = IceLoader(args=args, mode='train', modality=args.modality)
    val_set = IceLoader(args=args, mode='val', modality=args.modality)
    test_set = IceLoader(args=args, mode='test', modality=args.modality)

    ckpts_root = './save/{}/{}/{}/'.format(args.dataset, args.modality, args.classify)
    figs_root = './save/fig/{}/{}/{}/'.format(args.dataset, args.modality, args.classify)

    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    if not os.path.exists(figs_root):
        os.makedirs(figs_root)

    task_best_acc_list = []
    step_forgetting_list = []

    exemplar_class_vids = None
    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)

        train_set._update_exemplars(exemplar_class_vids)

        print('Incremental step: {}'.format(step))

        exemplar_class_vids, exemplar_class_mean = train(args, step, train_set, val_set, exemplar_class_vids)

        step_forgetting = detailed_test(args, step, test_set, task_best_acc_list)
        if step_forgetting is not None:
            step_forgetting_list.append(step_forgetting)
    Mean_forgetting = np.mean(step_forgetting_list)
    print('Average Forgetting: {:.6f}'.format(Mean_forgetting))
    
    if args.dataset != 'AVE' and args.modality != 'audio':
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()

