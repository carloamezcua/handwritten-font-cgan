from pathlib import Path

UPPERCASE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWERCASE = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
PUNCTUATION = list(".,!?()+-=/:;&")

CHARSET = UPPERCASE + LOWERCASE + DIGITS + PUNCTUATION
TEST_UNSEEN_CHARS = set("GQkz48?&+;)")

IMG_SIZE = 128
SEED = 42

CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARSET)}

# =============================================================================
# Hiperparámetros centralizados de entrenamiento (valores por defecto)
# Todos pueden sobreescribirse por CLI o desde Colab (--epochs, --lr, etc.).
# =============================================================================

JUDGES_CONFIG = {
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 7,
    "label_smoothing": 0.05,
}

PIX2PIX_CONFIG = {
    "lambda_l1": 100.0,
    "real_label": 0.9,
    "betas": (0.5, 0.999),
    "sample_every": 10,
    "modes": {
        "pretrain": {
            "data_dir": Path("data/synthetic_pairs"),
            "ckpt_name": "pix2pix_pretrained.pth",
            "epochs": 60,
            "batch_size": 32,
            "lr": 2e-4,
        },
        "scratch": {
            "data_dir": Path("data/real_pairs"),
            "ckpt_name": "pix2pix_scratch.pth",
            "epochs": 150,
            "batch_size": 16,
            "lr": 2e-4,
        },
        "finetune": {
            "data_dir": Path("data/real_pairs"),
            "ckpt_name": "pix2pix_finetuned.pth",
            "epochs": 100,
            "batch_size": 16,
            "lr": 1e-4,
        },
    },
}

STYLE_UNET_CONFIG = {
    "batch_size": 16,
    "style_dim": 128,
    "num_style_refs": 4,
    "lambda_l1": 100.0,
    "lambda_style": 5.0,
    "lambda_ink": 15.0,
    "betas": (0.5, 0.999),
    "sample_every": 10,
    "modes": {
        "pretrain": {
            "data_dir": Path("data/synthetic_pairs"),
            "ckpt_name": "style_unet_pretrained.pth",
            "epochs": 60,
            "lr": 2e-4,
        },
        "finetune": {
            "data_dir": Path("data/real_pairs"),
            "ckpt_name": "style_unet_finetuned.pth",
            "epochs": 40,
            "lr": 5e-5,
        },
    },
}
