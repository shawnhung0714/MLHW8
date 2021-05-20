from argparse import Namespace
from logging import fatal

config = Namespace(
    root_dir="data-bin",
    max_question_len=40,
    max_paragraph_len=250,
    doc_stride=75,
    batch_size=10000,
    logging_step=100,
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
