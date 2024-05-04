import omegaconf

from text.trainer import TextTrainer

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to yaml config file')
    args = parser.parse_args()

    cfg_path = args.config
    cfg = omegaconf.OmegaConf.load(cfg_path)

    trainer = TextTrainer(cfg)
    trainer.train()