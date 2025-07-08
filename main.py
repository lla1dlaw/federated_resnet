"""
Author: Liam Laidlaw
Purpose: A federated learning simulation environment for complex valued neural networks. This simulation was created for usage with a custom complex residual network. 

Based on the FLSim cifar10 example script available: https://github.com/facebookresearch/FLSim/blob/main/examples/cifar10_example.py
"""
import flsim.configs  # noqa
import hydra
import torch
from flsim.data.data_sharder import SequentialSharder
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    FLModel,
    MetricsReporter,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch


# custom Resnet Packages
from resnet.residual import (
    ComplexResNet,
    RealResNet,
    ComplexResidualBlock,
    RealResidualBlock,
)

from utils import (
    build_data_provider,
    send_email,
    save_model,
    save_results_to_csv,
)

def main(trainer_config, data_config, use_cuda_if_available: bool = True) -> None:
    # init model and setup cuda
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    model = ComplexResNet(
        architecture_type='WS',
        activation_function='crelu', 
        learn_imaginary_component=True, 
    )
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    # init trainer
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")
    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        examples_per_user=data_config.examples_per_user,
        drop_last=False,
    )

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1,
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="cifar10_tutorial")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    data_config = cfg.data

    main(
        trainer_config,
        data_config,
    )


def invoke_main() -> None:
    cfg = maybe_parse_json_config()
    run(cfg)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
