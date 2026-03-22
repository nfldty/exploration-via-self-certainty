from verl import DataProto
import math
import torch
from torch import nn
import ray
from verl.utils.reward_score import gsm8k, countdown, math_dataset
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif data_source == 'math':
        return math_dataset.compute_score
    else:
        raise NotImplementedError(f"Unknown data_source: {data_source}")


# ---------------------------------------------------------------------------
# RND network components (original iMENTOR intrinsic reward)
# ---------------------------------------------------------------------------

class RNDNet(nn.Module):
    def __init__(self, input_size, layers):
        super().__init__()
        self.input_size = input_size
        self.fn = nn.ModuleList([])
        self.layers = layers
        j = self.input_size
        for i in self.layers:
            self.fn.append(nn.Linear(j, i))
            self.fn.append(nn.LeakyReLU())
            j = i
        self.fn.append(nn.Linear(self.layers[-1], 1))
        self.fn.append(nn.Sigmoid())

    def forward(self, x):
        for func in self.fn:
            x = func(x)
        return x


class RNDReward(nn.Module):
    def __init__(self, input_size, layers, embedding):
        super().__init__()
        self.emb = nn.Embedding(embedding[0], embedding[1])
        self.input_size = input_size * embedding[1]
        self.target = RNDNet(self.input_size, layers)
        self.predictor = RNDNet(self.input_size, layers)
        self.bn0 = nn.BatchNorm1d(self.input_size)

    def forward(self, x):
        x = self.emb(x).view(-1, self.input_size)
        x = self.bn0(x)
        x_p = self.predictor(x)
        with torch.no_grad():
            x_t = self.target(x)
        x = ((x_t - x_p) ** 2)
        with torch.no_grad():
            x_min = x.min()
            x_max = x.max()
            x1 = 0.5 * (x - x_min) / (x_max - x_min)
        return x, x1


@ray.remote(num_gpus=1)
class RNDActor:
    def __init__(self, seq_len, layers, embedding, lr):
        self.device = torch.device("cuda:0")
        self.rnd_reward = RNDReward(seq_len, layers, embedding).to(self.device)
        self.rnd_optimizer = torch.optim.AdamW(self.rnd_reward.parameters(), lr=lr)

    def train_one_batch(self, rnd_inputs):
        rnd_0, rnd_1 = self.rnd_reward(rnd_inputs.to(self.device))
        rnd_1 = rnd_1.detach().cpu()
        loss_rnd = rnd_0.mean()
        self.rnd_optimizer.zero_grad()
        loss_rnd.backward()
        self.rnd_optimizer.step()
        return rnd_1


# ---------------------------------------------------------------------------
# Unified RewardManager with selectable intrinsic reward
# ---------------------------------------------------------------------------

class RewardManager():
    """Reward manager with configurable intrinsic reward type.

    Supported intrinsic_reward_type values:
      - 'self_certainty': z-score → sigmoid normalized self-certainty
      - 'rnd': Random Network Distillation (original iMENTOR)
      - 'none': no intrinsic reward (extrinsic only)
    """

    def __init__(self, tokenizer, num_examine, intrinsic_reward_type, scales,
                 rnd_trainer=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.intrinsic_reward_type = intrinsic_reward_type
        self.scales = list(scales)

        # RND-specific
        self.rnd_trainer = rnd_trainer

        # Self-certainty running statistics (Welford's algorithm)
        self._sc_count = 0
        self._sc_mean = 0.0
        self._sc_m2 = 0.0

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        intrinsic_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # ---- RND: batch-level forward pass ----
        rnd_per_sample = None
        if self.intrinsic_reward_type == 'rnd' and self.rnd_trainer is not None:
            rnd_inputs = data.batch["input_ids"] * data.batch["attention_mask"]
            rnd_per_sample = ray.get(self.rnd_trainer.train_one_batch.remote(rnd_inputs))

        # ---- Self-certainty: retrieve per-token values ----
        self_certainty = data.batch.get('self_certainty', None)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(
                error_score=0.0, format_score=0.1,
                solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            # Intrinsic reward (only for incorrect answers)
            if score < 1.0:
                if self.intrinsic_reward_type == 'rnd' and rnd_per_sample is not None:
                    intrinsic_reward_tensor[i, valid_response_length - 1] = (
                        rnd_per_sample[i] * self.scales[0] / self.scales[1])

                elif self.intrinsic_reward_type == 'self_certainty' and self_certainty is not None:
                    sc_tokens = self_certainty[i, :valid_response_length]
                    avg_sc = sc_tokens.mean().item()

                    self._sc_count += 1
                    delta = avg_sc - self._sc_mean
                    self._sc_mean += delta / self._sc_count
                    delta2 = avg_sc - self._sc_mean
                    self._sc_m2 += delta * delta2

                    std = math.sqrt(self._sc_m2 / self._sc_count) if self._sc_count > 1 else 1.0
                    # Negate: low self-certainty (more exploration) → large positive z → larger reward
                    z = (self._sc_mean - avg_sc) / max(std, 1e-8)
                    avg_sc_norm = 0.5 * (1.0 / (1.0 + math.exp(-z)))

                    intrinsic_reward_tensor[i, valid_response_length - 1] = (
                        avg_sc_norm * self.scales[0] / self.scales[1])

                # 'none': intrinsic_reward_tensor stays zero

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        print(sequences_str)
        self.scales[1] += self.scales[2]
        return reward_tensor, intrinsic_reward_tensor


class ValRewardManager():
    """Validation reward manager (no intrinsic reward)."""

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(
                error_score=0.0, format_score=0.0,
                solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # --- Select intrinsic reward type ---
    intrinsic_type = config.intrinsic_reward.type
    assert intrinsic_type in ('rnd', 'self_certainty', 'none'), \
        f"intrinsic_reward.type must be 'rnd', 'self_certainty', or 'none', got '{intrinsic_type}'"

    rnd_trainer = None
    if intrinsic_type == 'rnd':
        rnd_trainer = RNDActor.remote(
            seq_len=config.data.max_prompt_length + config.data.max_response_length,
            layers=config.imentor.layers,
            embedding=config.imentor.embedding,
            lr=config.imentor.lr)

    if intrinsic_type == 'self_certainty':
        scales = list(config.self_certainty.scales)
    elif intrinsic_type == 'rnd':
        scales = list(config.imentor.scales)
    else:
        scales = [0.0, 1.0, 0.0]

    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        intrinsic_reward_type=intrinsic_type,
        scales=scales,
        rnd_trainer=rnd_trainer)

    val_reward_fn = ValRewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
