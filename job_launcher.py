import sys
import os
import subprocess
import argparse
from datetime import datetime
import inspect
import shutil
import glob


from main_continual import str_to_dict

parser = argparse.ArgumentParser()
parser.add_argument("--script", type=str, required=True)
parser.add_argument("--mode", type=str, default="normal")
parser.add_argument("--experiment_dir", type=str, default=None)
parser.add_argument("--base_experiment_dir", type=str, default="./experiments")
parser.add_argument("--gpu", type=str, default="v100-16g")
parser.add_argument("--num_gpus", type=int, default=2)
parser.add_argument("--hours", type=int, default=20)
parser.add_argument("--requeue", type=int, default=0)

parser.add_argument("--linear_script", type=str, default=None)
parser.add_argument("--run_linear", action="store_true")

args = parser.parse_args()

# load file
if os.path.exists(args.script):
    with open(args.script) as f:
        command = [line.strip().strip("\\").strip() for line in f.readlines()]
else:
    print(f"{args.script} does not exist.")
    exit()

assert (
    "--checkpoint_dir" not in command
), "Please remove the --checkpoint_dir argument, it will be added automatically"

# collect args
command_args = str_to_dict(" ".join(command).split(" ")[2:])

# create experiment directory
if args.experiment_dir is None:
    args.experiment_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.experiment_dir += f"-{command_args['--name']}"
full_experiment_dir = os.path.join(args.base_experiment_dir, args.experiment_dir)
os.makedirs(full_experiment_dir, exist_ok=True) # Moved to main_continual.py
print(f"Experiment directory: {full_experiment_dir}")
shutil.copy(args.script, full_experiment_dir)
# add experiment directory to the command
command.extend(["--checkpoint_dir", full_experiment_dir])
command = " ".join(command)

print(command)

# run command
if args.mode == "normal":
    # ① pretrain 実行
    p = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stdout)
    retcode = p.wait()

    if retcode != 0:
        print(f"[ERROR] Pretrain job failed with return code {retcode}. Skip linear eval.")
        sys.exit(retcode)

    # ② linear eval 実行（オプション）
    if args.run_linear and args.linear_script is not None:
        print("[INFO] Pretrain finished. Starting linear evaluation for each task...")

        # main_continual の CLI から num_tasks / data_dir を取り出す
        num_tasks = int(command_args.get("--num_tasks", 1))
        data_dir = command_args.get("--data_dir", None)

        for task in range(num_tasks):
            # 対象タスクの ckpt を探索
            ckpt_pattern = os.path.join(full_experiment_dir, f"task{task}", "*.ckpt")
            ckpt_list = sorted(glob.glob(ckpt_pattern))
            if not ckpt_list:
                print(f"[WARN] No checkpoint found for task{task} in {ckpt_pattern}. Skipping.")
                continue

            ckpt = ckpt_list[-1]
            print(f"[INFO] Task {task}: using ckpt = {ckpt}")

            # main_linear.sh に渡す環境変数をセット
            env = os.environ.copy()
            env["TASK"] = str(task)
            if data_dir is not None:
                env["DATA_DIR"] = data_dir
            env["CKPT"] = ckpt

            linear_cmd = f"bash {args.linear_script}"
            print(f"[INFO] Running linear eval: {linear_cmd}")
            lp = subprocess.Popen(linear_cmd, shell=True,
                                  stdout=sys.stdout, stderr=sys.stdout, env=env)
            lp.wait()

        print("[INFO] All linear evaluations finished.")

elif args.mode == "slurm":
    # infer qos
    if 0 <= args.hours <= 2:
        qos = "qos_gpu-dev"
    elif args.hours <= 20:
        qos = "qos_gpu-t3"
    elif args.hours <= 100:
        qos = "qos_gpu-t4"

    # write command
    command_path = os.path.join(full_experiment_dir, "command.sh")
    with open(command_path, "w") as f:
        f.write(command)

    # run command
    p = subprocess.Popen(f"sbatch {command_path}", shell=True, stdout=sys.stdout, stderr=sys.stdout)
    p.wait()
