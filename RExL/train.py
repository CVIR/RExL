
import argparse
import sys, math
import os
import torch

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
        help="Sets the name of the dataset (PASCAL, MSCOCO, IMAGENET, PASCAL12).",
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
    parser.add_argument("-tl", '--tensorboard_log_dir', 
        help="Sets the path to the dataset.",
        default=None)
    parser.add_argument("-nt", "--num_timesteps",
        help="Sets the total no of timesteps.", type=int,
        default=2e6)
    parser.add_argument("-si", "--save_interval",
        help="Sets the timesteps after which model should be saved.", type=int,
        default=1e6)
    parser.add_argument("-sp", "--save_path",
        help="Path to save model.",
        default=None)
    parser.add_argument("-lp", "--load_path",
        help="Path to load model.",
        default=None)
    parser.add_argument("-vp", "--video_path",
        help="Path to video",
        default=None)

    parser.add_argument("-s", "--start",
        help="Start iteration",
        default=0)

    parser.add_argument("-i", "--idx",
        type=int,
        help="id of the image",
        default=-1)
    
        
    args = parser.parse_args()

    if args.root_path[-1] != '/':
        args.root_path = args.root_path + '/'

    if args.video_path is not None:
        args.video_path = args.video_path + '/'

    if args.save_path[-1] is not None and args.save_path[-1]!= '/':
        args.save_path = args.save_path + '/'

    if args.tensorboard_log_dir[-1] is not None and args.tensorboard_log_dir[-1] != '/':
        args.tensorboard_log_dir = args.tensorboard_log_dir + '/'

    if args.idx != -1:
        if args.save_path is not None:
            args.save_path = os.path.join(args.save_path, 'exp_' + args.model + '_im_' + str(args.idx) + '/')
        if args.tensorboard_log_dir is not None:
            args.tensorboard_log_dir = os.path.join(args.tensorboard_log_dir, 'exp_' + args.model + '_im_' + str(args.idx))
        if args.load_path is not None:
            args.load_path = os.path.join(args.load_path, 'exp_' + args.model + 'im_' + str(args.idx) + '/')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    

    from stable_baselines.common.policies import FeedForwardPolicy
    from stable_baselines.common import make_vec_env
    from stable_baselines import PPO2, ACER
    from stable_baselines.common.callbacks import CheckpointCallback

    import tensorflow as tf

    from explainableAI.envs import AttentionEnv

    from datetime import datetime


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
        "batch_size" : args.batch_size,
        "vid_path" : args.video_path,
        "idx" : args.idx, 
        "device" : torch.device("cuda:0")
    })


    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                            net_arch=[dict(pi=[256, 128],
                                                            vf=[256, 128])],
                                            feature_extraction="mlp", act_fun=tf.nn.relu)

    checkpoint_callback = CheckpointCallback(save_freq=args.save_interval, save_path=args.save_path, name_prefix='rl_model')
    
    model = None
    if args.load_path is None:
        model = ACER(CustomPolicy, vec_env, verbose=1, tensorboard_log=args.tensorboard_log_dir, gamma=1, n_steps=980, q_coef=1, rprop_alpha=0.9, learning_rate=0.00005)
    else:
        model = ACER.load(args.load_path)
        model.set_env(vec_env)

    num_steps = int(args.num_timesteps) - int(args.start)
    model.learn(total_timesteps=num_steps, tb_log_name='run', reset_num_timesteps=False, log_interval=10, callback=checkpoint_callback)

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    if args.save_path is not None:
        model.save(args.save_path + "%s_%s_%s_%s"%(args.model, str(args.kernel_size), args.dataset.lower(), date_time))

if __name__ == "__main__":
    main()
