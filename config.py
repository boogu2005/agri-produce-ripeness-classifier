import os


class Config:
    DATA_DIR = './'
    TRAIN_DIR = './train_dataset'
    VAL_DIR = './val_dataset'
    TEST_DIR = './test_dataset'
    OUTPUT_DIR = './results'
    TRAIN_RESULTS_DIR = './results/training'
    MODEL_DIR = './models'

    CLASSES = ['未熟', '半熟', '熟透']
    NUM_CLASSES = 3

    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    EPOCHS = 5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9

    DROPOUT_RATE = 0.5
    RANDOM_SEED = 42

    @classmethod
    def create_dirs(cls):
        for d in [cls.DATA_DIR, cls.TRAIN_DIR, cls.VAL_DIR, cls.TEST_DIR,
                  cls.OUTPUT_DIR, cls.MODEL_DIR]:
            os.makedirs(d, exist_ok=True)


config = Config()
