import os
from tqdm import tqdm
import torch
import einops
from PIL import Image
from pathlib import Path
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union, Any, Sequence, Literal, TypedDict
import torchvision.transforms.functional as transforms_f
import torch.nn.functional as F
import random
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.backend.observation import Observation
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode
from structures import (
    Observation,
    Demo,
    GripperPose,
    Instructions,
    Output,
    RotType,
    RotMode,
    Sample,
    EulerRotation,
    ContRotation,
    QuatRotation,
    Rotation,
    Position,
)
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
import cv2


Camera = Literal["wrist", "left_shoulder", "right_shoulder", "overhead", "front"]
Instructions = Dict[str, Dict[int, str]]


class Recorder(object):
    def __init__(self) -> None:
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        self.cam = VisionSensor.create([640, 320])
        self.cam.set_pose(np.array([1.35,0,1.58,-0.596,0.596, 0.38, -0.38]))
        # self.cam.set_pose(VisionSensor('cam_front').get_pose())
        self.cam.set_parent(cam_placeholder)
        self._snaps = []
        self._fps=30

    def take_snap(self):
        self._snaps.append(
            (self.cam.capture_rgb() * 255.).astype(np.uint8))
    
    def save(self, path):
        print('Converting to video ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'MJPG'), self._fps,
                tuple(self.cam.get_resolution()))
        for image in self._snaps:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()
        self._snaps = []
    def del_snap(self):
        self._snaps = []
        
class Mover:
    def __init__(self, task: TaskEnvironment, disabled: bool = False, max_tries: int = 1):
        self._task = task
        self._last_action: Optional[np.ndarray] = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action: np.ndarray, recorder: Recorder):
        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            obs, reward, terminate, other_obs = self._task.step(action, recorder)
            if other_obs == []:
                other_obs = [obs]
            for o in other_obs:
                images.append(
                    {
                        k.split("_")[0]: getattr(o, k)
                        for k in o.__dict__.keys()
                        if "_rgb" in k and getattr(o, k) is not None
                    }
                )

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())  # type: ignore
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())  # type: ignore
            # criteria = (dist_pos < 5e-2, dist_rot < 1e-1, (gripper > 0.5) == (target_gripper > 0.5))
            criteria = (dist_pos < 5e-2,)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            obs, reward, terminate, other_obs = self._task.step(action, recorder)
            if other_obs == []:
                other_obs = [obs]
            for o in other_obs:
                images.append(
                    {
                        k.split("_")[0]: getattr(o, k)
                        for k in o.__dict__.keys()
                        if "_rgb" in k and getattr(o, k) is not None
                    }
                )

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate, images

class Actioner:
    def __init__(
        self,
        record_actions: bool = False,
        replay_actions: Optional[Path] = None,
        ground_truth_rotation: bool = False,
        ground_truth_position: bool = False,
        ground_truth_gripper: bool = False,
        model: Optional[nn.Module] = None,  # model includes t and z
        model_rotation: Optional[nn.Module] = None,
        model_position: Optional[nn.Module] = None,
        model_gripper: Optional[nn.Module] = None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        instructions: Optional[Dict] = None,
        taskvar_token: bool = False,
        maxAction: int = 6,
    ):
        self._record_actions = record_actions
        self._replay_actions = replay_actions
        self._ground_truth_rotation = ground_truth_rotation
        self._ground_truth_position = ground_truth_position
        self._ground_truth_gripper = ground_truth_gripper
        assert (model is not None) ^ (
            model_rotation is not None
            and model_position is not None
            and model_gripper is not None
        )
        self._model = model
        self._model_rotation = model_rotation
        self._model_position = model_position
        self._model_gripper = model_gripper
        self._apply_cameras = apply_cameras
        self._instructions = instructions
        self._taskvar_token = taskvar_token
        self._maxAction = maxAction

        if self._taskvar_token:
            with open(Path(__file__).parent / "tasks.csv", "r") as fid:
                self._tasks = [l.strip() for l in fid.readlines()]

        self._actions: Dict = {}
        self._instr: List[str] = []
        self._taskvar: Optional[torch.Tensor] = None
        self._task: Optional[str] = None

    def load_episode(
        self, task_str: str, variation: int, demo_id: int, demo: Union[Demo, int]
    ):
        self._task = task_str

        if self._instructions is None:
            self._instr = None
        else:
            instructions = list(self._instructions[task_str][str(variation)])
            self._instr.clear()
            self._instr.append(random.choice(instructions))

        if self._taskvar_token:
            task_id = self._tasks.index(task_str)
            self._taskvar = torch.Tensor([[task_id, variation]]).unsqueeze(0)
            print(self._taskvar)

        if self._replay_actions is not None:
            self._actions = torch.load(
                self._replay_actions / f"episode{demo_id}" / "actions.pth"
            )
        elif (
            self._ground_truth_rotation
            or self._ground_truth_position
            or self._ground_truth_gripper
        ):
            if isinstance(demo, int):
                raise NotImplementedError()
            action_ls = self.get_action_from_demo(demo)
            self._actions = dict(enumerate(action_ls))
        else:
            self._actions = {}

    def get_action_from_demo(self, demo: Demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        action_ls = []
        for f in key_frame:
            obs = demo[f]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])  # type: ignore
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))
        return action_ls

    def predict(
        self, step_id: int, rgbs: torch.Tensor, pcds: torch.Tensor, gripper: torch.Tensor = None
    ) -> Dict[str, Any]:
        
        T = rgbs.shape[1]
        
        # pad_len = (self._maxAction - T)
        # img_pad_vec = [0, 0] * rgbs.dim()
        # img_pad_vec[-3] = pad_len
        # rgbs = F.pad(rgbs, img_pad_vec, value=0)
        # pcds = F.pad(pcds, img_pad_vec, value=0)

        # padding_mask = torch.tensor([True] * T + [False] * pad_len)
        padding_mask = torch.tensor([True] * T)
        padding_mask = padding_mask.unsqueeze(0)

        # rgbs = einops.repeat(rgbs, "b t c h w -> (repeat b) t c h w", repeat=16)

        # padding_mask = torch.ones_like(pcds[:, :,0, 0, 0]).bool()
        output: Dict[str, Any] = {"action": None, "attention": {}}

        if self._instr is not None:
            self._instr = self._instr

        if self._taskvar is not None:
            self._taskvar = self._taskvar.to(rgbs.device)

        if self._replay_actions:
            if step_id not in self._actions:
                print(f"Step {step_id} is not prerecorded!")
                return output
            action = self._actions[step_id]
        elif self._model is None:
            action = torch.Tensor([]).to(self.device)
            keys = ("position", "rotation", "gripper")
            slices = (slice(0, 3), slice(3, 7), slice(7, 8))
            for key, slice_ in zip(keys, slices):
                model = getattr(self, f"_model_{key}")
                t = model["t"][self._task][: step_id + 1].unsqueeze(0)
                z = model["z"][self._task][: step_id + 1].unsqueeze(0)
                pred = model["model"](
                    rgbs, pcds, padding_mask, t, z, self._instr, gripper, self._taskvar
                )
                action_key = model["model"].compute_action(pred)
                action = torch.cat([action, action_key[slice_]])
            output["action"] = action
        else:
            if self._task is None:
                raise ValueError()

            z_offset = self._model["model"].z_dict[self._task][: step_id + 1].unsqueeze(0)
            pred = self._model["model"](rgbs, self._instr, pcds, padding_mask, z_offset)
            
            # pred = self._model["model"](
            #     rgbs, pcds, padding_mask, t, z, self._instr, gripper, self._taskvar
            # )
            # pred = self._model["model"](rgbs, self._instr, pcds, padding_mask)

            output["action"] = pred["action"]  # type: ignore
            output["attention"] = pred["attention"]

        if self._ground_truth_rotation:
            if step_id not in self._actions:
                print(f"No ground truth available for step {step_id}!")
                return output
            output["action"][:, 3:7] = self._actions[step_id][:, 3:7]
        if self._ground_truth_position:
            if step_id not in self._actions:
                print(f"No ground truth available for step {step_id}!")
                return output
            output["action"][:, :3] = self._actions[step_id][:, :3]
        if self._ground_truth_gripper:
            if step_id not in self._actions:
                print(f"No ground truth available for step {step_id}!")
                return output
            output["action"][:, 7] = self._actions[step_id][:, 7]

        if self._record_actions:
            self._actions[step_id] = output["action"]

        return output

    def save(self, ep_dir):
        if self._record_actions:
            torch.save(self._actions, ep_dir / "actions.pth")

    @property
    def device(self):
        if self._model is not None:
            return next(self._model["model"].parameters()).device
        return next(self._model_position["model"].parameters()).device  # type: ignore

    def eval(self):
        if self._model is not None:
            self._model["model"].eval()
        else:
            self._model_position["model"].eval()  # type: ignore
            self._model_rotation["model"].eval()  # type: ignore
            self._model_gripper["model"].eval()  # type: ignore


class RLBenchEnv_RT1:
    def __init__(
        self,
        data_path,
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        image_size=(128,128),
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        gripper_pose: GripperPose = "none",
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.gripper_pose = gripper_pose

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            apply_rgb, apply_depth, apply_pc, apply_cameras, image_size
        )
        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config, headless=headless
        )

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(
        self, obs: Observation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2,
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        rgb = transforms_f.normalize(
            rgb.float(), 
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )
        
        if "attn" in self.gripper_pose:
            attns = torch.Tensor([])
            for cam in self.apply_cameras:
                u, v = obs_to_attn(obs, cam)
                attn = torch.zeros((1, 1, 1, 128, 128))
                if not (u < 0 or u > 127 or v < 0 or v > 127):
                    attn[0, 0, 0, v, u] = 1
                attns = torch.cat([attns, attn], 1)
            rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

    def get_obs_action_from_demo(self, demo: Demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False,
        )
        return demos

    def evaluate(
        self,
        task_str: str,
        max_episodes: int,
        variation: int,
        num_demos: int,
        log_dir: Optional[Path],
        actioner: Actioner,
        offset: int = 0,
        max_tries: int = 10,
        demos: Optional[List[Demo]] = None,
        save_attn: bool = False,
        save_video: bool = False,
    ):
        """
        Evaluate the policy network on the desired demo or test environments
            :param task_type: type of task to evaluate
            :param max_episodes: maximum episodes to finish a task
            :param num_demos: number of test demos for evaluation
            :param model: the policy network
            :param demos: whether to use the saved demos
            :return: success rate
        """

        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task.set_variation(variation)  # type: ignore

        actioner.eval()
        device = actioner.device

        success_rate = 0.0

        if demos is None:
            fetch_list = [i for i in range(num_demos)]
        else:
            fetch_list = demos

        fetch_list = fetch_list[offset:]
        
        if save_video:
            recorder = Recorder()

        with torch.no_grad():
            for demo_id, demo in enumerate(tqdm(fetch_list)):

                rgbs = torch.Tensor([]).to(device)
                pcds = torch.Tensor([]).to(device)
                grippers = torch.Tensor([]).to(device)

                # reset a new demo or a defined demo in the demo list
                if isinstance(demo, int):
                    _, obs = task.reset()
                else:
                    print("Resetting to demo")
                    print(demo)
                    _, obs = task.reset_to_demo(demo)  # type: ignore

                actioner.load_episode(task_str, variation, demo_id, demo)

                move = Mover(task, max_tries=max_tries)
                reward = None

                for step_id in range(max_episodes):
                    # fetch the current observation, and predict one action
                    rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)  # type: ignore

                    rgb = rgb.to(device)
                    pcd = pcd.to(device)

                    rgbs = torch.cat([rgbs, rgb.unsqueeze(0)], dim=1)
                    pcds = torch.cat([pcds, pcd.unsqueeze(0)], dim=1)
                    # grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)
                    output = actioner.predict(step_id, rgbs, pcds)
                    action = output["action"]

                    if action is None:
                        break

                    if log_dir is not None and save_attn and output["action"] is not None:
                        ep_dir = log_dir / f"episode{demo_id}"
                        fig = plot_attention(
                            output["attention"][-1],
                            rgbs[0][-1, :, :3],
                            pcds[0][-1].view(3, 3, 128, 128),
                            ep_dir / f"attn_{step_id}.png",
                        )
                        attention = output["attention"][-1][0].squeeze(0).cpu().numpy().astype(np.uint8)
                        Image.fromarray(attention * 255, mode='L').save(ep_dir / f"attention.png")

                    # update the observation based on the predicted action
                    try:
                        action_np = action[-1].detach().cpu().numpy()
                        
                        if save_video:
                            obs, reward, terminate, _ = move(action_np, recorder)
                        else:
                            obs, reward, terminate, step_images = move(action_np, None)

                        if reward == 1:
                            success_rate += 1 / num_demos
                            break
                        if terminate:
                            print("The episode has terminated!")
                    except (IKError, ConfigurationPathError, InvalidActionError) as e:
                        print(task_type, demo, step_id, success_rate, e)
                        reward = 0
                        break
                if save_video: 
                    recorder.save(f"./records/{task_str}_{demo_id}.avi")
                    recorder.del_snap()
                print(
                    task_str,
                    "Reward",
                    reward,
                    "Variation",
                    variation,
                    "Step",
                    demo_id,
                    "SR: %.2f" % (success_rate * 100),
                )

                # if log_dir is not None:
                #     ep_dir = log_dir / task_str / f"episode{demo_id}"
                #     ep_dir.mkdir(exist_ok=True, parents=True)
                #     for frame_id, img_by_cam in enumerate(images):
                #         for cam, im in img_by_cam.items():
                #             cam_dir = ep_dir / cam
                #             cam_dir.mkdir(exist_ok=True, parents=True)
                #             Image.fromarray(im).save(cam_dir / f"{frame_id}.png")

        self.env.shutdown()
        return success_rate

    def create_obs_config(
        self, apply_rgb, apply_depth, apply_pc, apply_cameras, image_size, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            image_size=image_size,
            mask=False,
            render_mode=RenderMode.OPENGL,
            **kwargs,
        )

        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config

def plot_attention(
    attentions: torch.Tensor, rgbs: torch.Tensor, pcds: torch.Tensor, dest: Path
) -> plt.Figure:
    attentions = attentions.detach().cpu()
    rgbs = rgbs.detach().cpu()
    pcds = pcds.detach().cpu()

    ep_dir = dest.parent
    ep_dir.mkdir(exist_ok=True, parents=True)
    name = dest.stem
    ext = dest.suffix

    # plt.figure(figsize=(10, 8))
    num_cameras = len(attentions)
    for i, (a, rgb, pcd) in enumerate(zip(attentions, rgbs, pcds)):
        # plt.subplot(num_cameras, 4, i * 4 + 1)
        plt.imshow(a.permute(1, 2, 0).log())
        plt.axis("off")
        plt.colorbar()
        plt.savefig(ep_dir / f"{name}-{i}-attn{ext}", bbox_inches="tight")
        plt.tight_layout()
        plt.clf()

        # plt.subplot(num_cameras, 4, i * 4 + 2)
        # plt.imshow(a.permute(1, 2, 0))
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.clf()

        # plt.subplot(num_cameras, 4, i * 4 + 3)
        plt.imshow(((rgb + 1) / 2).permute(1, 2, 0))
        plt.axis("off")
        plt.savefig(ep_dir / f"{name}-{i}-rgb{ext}", bbox_inches="tight")
        plt.tight_layout()
        plt.clf()

        # pcd_norm = (pcd - pcd.min(0).values) / (pcd.max(0).values - pcd.min(0).values)
        # # plt.subplot(num_cameras, 4, i * 4 + 4)
        # plt.imshow(pcd_norm.permute(1, 2, 0))
        # plt.axis("off")
        # plt.savefig(ep_dir / f"{name}-{i}-pcd{ext}", bbox_inches="tight")
        # plt.tight_layout()
        # plt.clf()

    # pcd_norm = (pcds - pcds.min(0).values) / (pcds.max(0).values - pcds.min(0).values)
    # # plt.subplot(num_cameras, 4, i * 4 + 4)
    # plt.imshow(pcd_norm.permute(1, 2, 0))
    # plt.axis("off")
    # plt.savefig(ep_dir / f"{name}-pcd{ext}", bbox_inches="tight")
    # plt.tight_layout()
    # plt.clf()

    return plt.gcf()


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def _is_stopped(demo, i, obs, stopped_buffer):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped

def keypoint_discovery(demo: Demo) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints

def obs_to_attn(obs, camera: str) -> Tuple[int, int]:
    extrinsics_44 = torch.from_numpy(obs.misc[f"{camera}_camera_extrinsics"]).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(obs.misc[f"{camera}_camera_intrinsics"]).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)


def tokenize_act_values(x, bins=256, act_continuous=True, shift=32000):
    # Appendix B. Agent Data Tokenization Details
    # 动作向量  (B, T, _) 已经位于 [-1, 1] 区间内
    if act_continuous:
        c = x
        #  使用 1024 bins 离散化 并 shift 结果整数（到[32000,33024]）
        c = (c + 1) * (bins / 2)
        c = c.type(torch.int32)
        c += shift
    else:
        c = x + shift
    return c


def inverse_tokenize_act_values(x, bins=256, act_continuous=True, shift=32000):
    # 从 tokenize 的动作返回原动作
    if act_continuous:
        c = x - shift
        c = (2 * c) / bins - 1
        c = c.type(torch.float32)
    else:
        c = x - shift
    return c




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Sample(TypedDict):
    frame_id: torch.Tensor
    task: Union[List[str], str]
    variation: Union[List[int], int]
    rgbs: torch.Tensor
    pcds: torch.Tensor
    action: torch.Tensor
    padding_mask: torch.Tensor
    # instr: torch.Tensor
    gripper: torch.Tensor


def get_log_dir(xp) -> Path:
    log_dir = xp
    version = int(os.environ.get("SLURM_JOBID", 0))
    while (log_dir / f"version{version}").is_dir():
        version += 1
    return log_dir / f"version{version}"

class CheckpointCallback:
    def __init__(
        self,
        log_dir: Path,
        state_dict: Any,
    ):
        self._log_dir = log_dir / "checkpoints"
        self._state_dict = state_dict

        if os.path.exists(self._log_dir):
            pass
        else:
            os.makedirs(self._log_dir)

    def __call__(self, step):

        dest = self._log_dir / f"epoch={step}.pth"
        torch.save(self._state_dict, dest)


def compute_rotation_loss(logit: torch.Tensor, rot: torch.Tensor):
    dtype = logit.dtype

    rot_ = -rot.clone()

    loss = F.mse_loss(logit, rot, reduction="none").to(dtype)
    loss = loss.mean(1)

    loss_ = F.mse_loss(logit, rot_, reduction="none").to(dtype)
    loss_ = loss_.mean(1)

    select_mask = (loss < loss_).float()

    sym_loss = 4 * (select_mask * loss + (1 - select_mask) * loss_)

    return {"rotation": sym_loss.mean()}

def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)

def normalise_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


def compute_rotation_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    pred = norm_tensor(pred)
    acc = (pred - true).abs().max(1).values < 0.05
    acc = acc.to(pred.dtype)

    if reduction == "mean":
        acc = acc.mean()
    return {"rotation": acc}


class Loss_Metrics:
    def __init__(
        self,args
    ):
        self.tasks = list(args.tasks)

    def compute_loss(
        self, pred: torch.Tensor, sample: Sample
    ) -> Dict[str, torch.Tensor]:
        device = pred.float().device
        padding_mask = sample["padding_mask"].to(device)    # (B, T)
        outputs = sample["action"].to(device)[padding_mask].float() # (B, T, D)
        # pred = pred[padding_mask]

        # outputs = tokenize_act_values(outputs)

        losses = {}
        losses["position"] = F.mse_loss(pred[:, :3], outputs[:, :3]) * 3
        
        losses.update(compute_rotation_loss(pred[:, 3:7].float(), outputs[:, 3:7]))
        losses["gripper"] = F.mse_loss(pred[:, 7:8].float(), outputs[:, 7:8])
        # if pred["task"] is not None:    # torch.Size([8, 106])
        #     task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
        #     task = task.to(device).long()
        #     losses["task"] = F.cross_entropy(pred["task"].float(), task)
        return losses

    def compute_metrics(
        self, pred: torch.Tensor, sample: Sample
    ) -> Dict[str, torch.Tensor]:
        device = pred.device
        dtype = pred.dtype
        padding_mask = sample["padding_mask"].to(device)
        outputs = sample["action"].to(device)[padding_mask]
        # pred = pred[padding_mask]

        # pred = inverse_tokenize_act_values(pred)
        # outputs = inverse_tokenize_act_values(outputs)
        
        metrics = {}
        acc = ((pred[:, :3] - outputs[:, :3]) ** 2).sum(1).sqrt() < 0.01
        metrics["position"] = acc.to(dtype).mean()

        pred_gripper = (pred[:, 7:8] > 0.5).squeeze(-1)
        true_gripper = outputs[:, 7].bool()
        acc = pred_gripper == true_gripper
        metrics["gripper"] = acc.to(dtype).mean()

        metrics.update(compute_rotation_metrics(pred[:, 3:7], outputs[:, 3:7]))

        # task = torch.Tensor([self.tasks.index(t) for t in sample["task"]])
        # task = task.to(device).long()
        # acc = task == pred["task"].argmax(1)
        # metrics["task"] = acc.to(dtype).mean()

        return metrics