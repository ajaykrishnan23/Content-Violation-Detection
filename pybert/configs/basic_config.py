
from pathlib import Path
BASE_DIR = Path('pybert')
config = {
    'writer_dir': BASE_DIR / "output/TSboard",
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'cache_dir': BASE_DIR / 'model/',
    'result': BASE_DIR / "output/result",

    'bert_vocab_path': BASE_DIR / 'pretrain/bert/base-uncased/bert_vocab.txt',
    'bert_config_file': BASE_DIR / 'pretrain/bert/base-uncased/config.json',
    'bert_model_dir': BASE_DIR / 'pretrain/bert/base-uncased',

}

