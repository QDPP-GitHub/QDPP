import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str, time_spent
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    args.results_path = args.results_path + '/' + str(_run._id)
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(args.results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    
    # print('_run', _run.__dict__)
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    to_index_flag = False
    if hasattr(args, 'to_index_flag'):
        if args.to_index_flag:
            to_index_flag = True
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    # if args.disc_state:
    #     if args.env_args["map_name"] == '3m':
    #         state_num = 1077
    #         if to_index_flag:
    #             pass
    #         else:
    #             state_shape = state_num
    #     elif args.env_args["map_name"] == 'corridor':
    #         state_num = 5280
    #         if to_index_flag:
    #             pass
    #         else:
    #             state_shape = state_num
    #     elif args.env_args["map_name"] == '6h_vs_8z':
    #         state_num = 2884
    #         if to_index_flag:
    #             state_shape = 62
    #         else:
    #             state_shape = state_num
    #     elif args.env_args["map_name"] == '2s3z':
    #         state_num = 2325
    #         # state_num = 165
    #         if to_index_flag:
    #             state_shape = 20
    #         else:
    #             state_shape = state_num
    #     else:
    #         raise NotImplementedError
    # else:
    state_shape = env_info["state_shape"]
    state_num = env_info.get("state_num", None)
    # TEST
    # if args.env_args["map_name"] == '2s3z':
    #     state_shape = 120
    #     state_num = state_shape
    # state_shape = env_info["state_shape"]
    # state_num = env_info.get("state_num", None)
    # state_num = state_shape
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = state_shape
    args.state_num = state_num
    args.all_obs = env_info.get("all_obs", None)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": state_shape},
        # "state": {"vshape": state_num},  # TEST
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "noise": {"vshape": (args.noise_dim,)}
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
        runner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # min_training_interval
    # training_interval_count = 0.0
    # episode_limit = env_info["episode_limit"]
    last_train_T = -env_info["episode_limit"] - 1
    # args.env_args.episode_limit
    # train_intervel_step = 0
    training_times = 0

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        time_stamp = time.time()

        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        time_stamp = time_spent(time_stamp, 'Sampling')

        if buffer.can_sample(args.batch_size):
            if (runner.t_env - last_train_T) / env_info["episode_limit"] >= 0.9:
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                training_times += 1
                logger.console_logger.info("t_env: {} / training_times {}".format(runner.t_env, training_times))
                # print('training_times', training_times)
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                time_stamp = time.time()

                learner.train(episode_sample, runner.t_env, episode)
                last_train_T = runner.t_env

                time_stamp = time_spent(time_stamp, 'Training')

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

            if args.noise_bandit:
                for _ in range(n_test_runs):
                    runner.run(test_mode=True, test_uniform=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)
            runner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config