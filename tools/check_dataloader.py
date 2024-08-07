from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

from pcdet.utils import common_utils


def main():
    cfg_file = "tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml"

    cfg_from_yaml_file(cfg_file, cfg)
    
    dist_train = False

    logger = common_utils.create_logger('log.txt', rank=cfg.LOCAL_RANK)

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=12,
        dist=dist_train, workers=2,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=True,
        total_epochs=4,
        seed=666
    )

    print("-> Done creating dataloader")
    print(type(train_set))
    print(f"{len(train_set.infos)=}")
    print(f"{train_set.camera_config=}")
    print(f"{train_set._merge_all_iters_to_one_epoch=}")
    print(train_set.__getitem__(0))
