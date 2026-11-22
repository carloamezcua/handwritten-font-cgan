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
