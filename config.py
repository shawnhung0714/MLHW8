from argparse import Namespace

config = Namespace(
    root_dir="data-bin",
    batch_size=10000,
    save_path="models",
    optim_hparas={"lr": 1e-3,},
    n_epochs=50,
    # early_stop=100,
    num_workers=4,
    resume_model=False,
    use_wandb=True,
    model_type="cnn",
    out_file="PREDICTION_FILE.csv"
)
