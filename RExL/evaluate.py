import argparse
import matplotlib.pyplot as plt 
import sys
import os
import csv
import matplotlib as mpl
from PIL import Image

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-ne", "--num_env",
        help="Sets the no of the environments to run in parallel.", type=int,
        default=1)
    parser.add_argument("-m", "--model",
        help="Select which model to use (vgg, resnet).",
        default="vgg")
    parser.add_argument("-ks","--kernel_size", 
        help="Sets the kernel size for insertion or deletion.", type=int,
        default=7)
    parser.add_argument("-rt", "--refresh_time", 
        help="Sets the number of times a single image needs to be reused.", type=int,
        default=1)
    parser.add_argument("-ci", "--class_index", 
        help="Sets which class to train on. If not set trains on all classes in the dataset.", type=int,
        default=-1)
    parser.add_argument("-nc", '--no_of_training_classes', 
        help="Sets the no of training classes.", type=int,
        default=None)
    parser.add_argument("-rp", '--root_path', 
        help="Sets the path to the dataset.",
        default=None)
    parser.add_argument("-d", '--dataset', 
        help="Sets the name of the dataset (PASCAL, MSCOCO, IMAGENET).",
        default='PASCAL')
    parser.add_argument("-dt", '--dataset_type', 
        help="Sets the path to the dataset type (train, val).",
        default='train')
    parser.add_argument("-mt", '--mode_of_training', 
        help="Sets the mode of training(insertion, deletion).",
        default='deletion')
    parser.add_argument("-bs", '--batch_size', 
        help="Sets the batch size of the dataloader.", type=int,
        default=1)
    parser.add_argument("-v", '--verbose', 
        help="Sets the verbose(0 no display, 1 for visualization)", type=int,
        default=0)
    parser.add_argument("-lp", "--load_path",
        help="Path to load model.",
        default=None)
    parser.add_argument("-i", "--idx",
        help="id of the image",
        default=-1)
    parser.add_argument("-ip", "--image_path",
        help="path to store image",
        default=None)
    parser.add_argument("-log", "--log_path",
        help="log_path",
        default=None)
        
    args = parser.parse_args()

    if args.root_path[-1] != '/':
        args.root_path = args.root_path + '/'
    
    if args.image_path is not None and args.root_path[-1] != '/':
        args.root_path = args.root_path + '/'

    log_file_name = 'log_0.csv'
    log_path = None
    if args.log_path is not None:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)

    if args.log_path is not None:
        log_path = os.path.join(args.log_path, log_file_name)

        f = open(log_path, 'a')
        writer = csv.writer(f, delimiter=',')
    
    print(args.idx)

    if int(args.idx) != -1:

        if args.load_path is not None:
            args.load_path = os.path.join(args.load_path, 'exp_' + args.model + '_im_' + str(args.idx) + '/')
        if args.image_path is not None:
            args.image_path = os.path.join(args.image_path, 'exp_' + args.model + '_im_' + str(args.idx) + '/')

    if args.image_path is not None and not os.path.exists(args.image_path):
        os.makedirs(args.image_path)

    print('Load path: ', args.load_path)
    if args.load_path is not None and os.path.isdir(args.load_path):
        models = os.listdir(args.load_path)
        args.load_path = os.path.join(args.load_path, models[-1])

    print(args.load_path)


    
    from stable_baselines.common import make_vec_env
    # from stable_baselines import DQN
    from stable_baselines import PPO2, ACER
    
    import torch 
    import numpy as np
    from skimage.transform import resize

    from explainableAI.envs import AttentionEnv
    
    from RISE.evaluation import CausalMetric, auc, n_auc, gkern
    from tqdm import tqdm

    vec_env = make_vec_env(AttentionEnv, n_envs=args.num_env, env_kwargs={
        "model_name" : args.model,
        "kernelSize" : args.kernel_size, 
        "refresh_time" : args.refresh_time, 
        "class_index" : args.class_index, 
        "no_of_training_classes" : args.no_of_training_classes, 
        "RootPath" : args.root_path, 
        "datasetName" : args.dataset, 
        "datasetType" : args.dataset_type, 
        "mode_of_training" : args.mode_of_training,
        "idx" : args.idx,
        "device" : torch.device("cuda:0")
    })
    
    if args.verbose and args.batch_size > 1:
        print("Not possible, for visualization, keep the batch size == 1")
        return
    
    model = ACER.load(args.load_path)
    
    mean_ins_auc = 0
    mean_del_auc = 0
    
    if args.dataset == 'IMAGENET' or args.dataset == 'PASCAL12': 
        sub_fn = lambda x : (torch.rand_like(x) - 0.5)
    else:
        sub_fn = lambda x : (torch.rand_like(x) - 0.5)*255

    # sub_fn = torch.zeros_like
    klen =11
    ksig = 5
    blur_kern = gkern(klen, ksig)
    # self.blur = lambda x: nn.functional.conv2d(x, blur_kern, padding=klen//2)
    sub_fn_ins = lambda x: torch.nn.functional.conv2d(x, blur_kern, padding=klen//2)
    
    dataloader = vec_env.get_attr('dataLoader')[0]
    n_classes = vec_env.get_attr('no_of_training_classes')[0] 
    mode_of_training = vec_env.get_attr('mode_of_training')[0]
    kernel_size = vec_env.get_attr('kernelSize')[0]
    bb_model = vec_env.get_attr('model')[0]
    device = vec_env.get_attr('device')[0]

    n_classes = 1000
    
    insertion = CausalMetric(bb_model, 'ins', 224*8, substrate_fn=sub_fn_ins, n_classes=n_classes, device=device)
    deletion = CausalMetric(bb_model, 'del', 224*8, substrate_fn=sub_fn, n_classes=n_classes, device=device)
    
    images_stack = None
    explanations_stack = None
    stack_counter = 0
    num_evaluations = 0
    dataset_size = len(dataloader)
    obs = vec_env.reset()
    for i in tqdm(range(dataset_size)):

        # if i >=1:
        #     break
        # if i < 4:
        #     obs = vec_env.reset()
        #     continue

        # if i >= 5:
        #     break

        image = vec_env.get_attr('canvas')[0] if mode_of_training == 'deletion' else vec_env.get_attr('currentImage')[0]
        img_copy = image.clone().cpu()
        img_path = vec_env.get_attr('img_path')[0]
        img_cpu = image.clone().cpu().numpy()[0]
        
        dones = [False]
        if args.mode_of_training == 'deletion':
            max_pred = vec_env.get_attr('max_pred')[0]
            old_reward = -max_pred
        else:
            old_reward = 0
        sal = np.zeros((kernel_size, kernel_size))
        uncov = np.zeros((kernel_size, kernel_size))

        s = 0
        while dones[0] == False:
            s += 1
            lam = 1
            uncov *= lam

            action, _states = model.predict(obs)

            obs, rewards, dones, info = vec_env.step(action)
            if args.mode_of_training == 'insertion':
                rewards[0] += 1
            uncov[action[0] // kernel_size, action[0] % kernel_size] = 1
            sal += (rewards[0] - old_reward)*uncov
            if vec_env.get_attr('step_count')[0] == kernel_size*kernel_size:
                dones[0] = True
            old_reward = rewards[0]
        if args.mode_of_training == 'deletion':
            sal = ( sal - np.min(sal) ) / (np.max(sal))
        else:
            sal = sal /np.max(sal)
        explanation = resize(sal, (224, 224), order=1, mode='reflect', anti_aliasing=False)
        
        ######## Display the saliencies ##################
        if args.verbose:
            if(args.dataset == 'IMAGENET' or args.dataset == 'PASCAL12'):
                
                mean = [0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]

                for channel in range(3):
                    img_cpu[channel] = img_cpu[channel]*std[channel] + mean[channel]

                img_cpu = img_cpu*255
                # img_cpu = img_cpu[(2,1,0),:,:]      
                
            else:
                mean = [104.01, 116.67, 122.68]
                for channel in range(3):
                    img_cpu[channel] = img_cpu[channel] + mean[channel]

                img_cpu = img_cpu[(2,1,0),:,:]
                
            img_cpu = img_cpu.transpose((1,2,0))
            img_cpu[img_cpu < 0] = 0

            cm = mpl.cm.get_cmap('jet')
            explanation_ = cm(explanation)
            explanation_ = explanation_[:, :, :3]
            explanation_ *=255
            final_img = explanation_ * 0.5 + img_cpu * 0.5
            final_img = final_img.astype(np.uint8)
            final_img = Image.fromarray(final_img)

            final_img.save(args.image_path + '/' + str(i) + '.jpg')
        
        ####################################

        if args.batch_size == 1:    
            scores2 = deletion.single_run(img_copy, explanation, verbose=0, save_to = None)
            scores1 = insertion.single_run(img_copy, explanation, verbose=0, save_to = None)
            num_evaluations += 1
            stack_counter = 1

            writer.writerow([img_path, auc(scores1), auc(scores2)])
            mean_ins_auc += auc(scores1)
            mean_del_auc += auc(scores2)
            print("------------Summary (so far)----------------------------", flush=True)
            print("image: {}".format(i), flush = True)
            print("insertion score : ", auc(scores1), flush = True)
            print("deletion score", auc(scores2), flush = True)
            print("--------------------------------------------------------", flush = True)
         
        else:
            if stack_counter == args.batch_size:
                scores2 = deletion.evaluate(images_stack, explanations_stack, batch_size = args.batch_size)
                scores1 = insertion.evaluate(images_stack, explanations_stack, batch_size=args.batch_size)
                
                images_stack = img_copy
                explanations_stack = explanation

                mean_ins_auc += auc(scores1.mean(1))
                mean_del_auc += auc(scores2.mean(1))
                stack_counter = 1
            
            else:
                if images_stack is None:
                    images_stack = img_copy
                    explanations_stack = explanation

                else:
                    images_stack = torch.cat([images_stack, img_copy], dim = 0)
                    explanations_stack = np.vstack((explanations_stack, explanation))
                    
                stack_counter += 1
                
        num_evaluations = int((i+1)/args.batch_size)

        if num_evaluations%1 == 0 and stack_counter == 1 and num_evaluations > 0:
            print("------------Summary (so far)----------------------------", flush=True)
            print("Number of evaluations: {} and number of images: {}".format(num_evaluations, i), flush = True)
            print("insertion score : ", mean_ins_auc/num_evaluations, flush = True)
            print("deletion score", mean_del_auc/num_evaluations, flush = True)
            print("--------------------------------------------------------", flush = True)
        

    print("ins score : ", mean_ins_auc/num_evaluations, flush = True)
    print("del score", mean_del_auc/num_evaluations, flush = True)


if __name__ == "__main__":
    main()
