import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset

@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    trainloaders,validationloaders,testloaders = prepare_dataset(cfg.num_clients, cfg.config_fit.batch_size)
if __name__ == "__main__":
    main()