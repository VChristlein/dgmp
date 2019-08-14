from datetime import datetime

# dictionaries contain global variables and paths for I/O depending on the environment

_current_date = datetime.today().strftime('%Y-%m-%d')

WRITER_DATA_DIR = {'default': 'data/writer_new',
                   'cluster': '/cluster/im86amuv/data/writer_split',
                   'gcloud': '/home/luspr/writer-id/data'}

ICDAR2013_DIR = {
    'default': 'data/icdar2013',
    'cluster': '/cluster/im86amuv/data/icdar2013'
}

RANKINGS_OUTPUT_DIR = {'default': 'rankings/{}'.format(_current_date),
                       'cluster': '/home/im86amuv/outputs/rankings'}

MODEL_OUTPUT_DIR = {'default': 'models',
                    'cluster': 'models'}

MEAN_WRITER = [0.796997, 0.741427, 0.643161]

STD_WRITER = [0.123953, 0.129172, 0.134826]

CONTEXT = 'default'

# Denotes whether the images are colored, i.e. three channels, (= True) or are binarized (= False)
COLOR = True
USE_PATCHES = False

""" Default values """

TRAIN_BATCH_SIZE = 25
GUARANTEED_TRIPLETS = 12

DEFAULT_EPOCHS = 35
DEFAULT_LR = 1e-4

DEFAULT_LAMBDA = 1.0
DEFAULT_MARGIN = 0.1

DEFAULT_LAMBDA_START = 1.0
DEFAULT_LAMBDA_END = 1e9
DEFAULT_LAMBDA_MULTIPLIER = 100

DEFAULT_LR_START = 1e-3
DEFAULT_LR_END = 1e-7
DEFAULT_LR_MULTIPLIER = 0.1

DEFAULT_NUM_CLASSES = 6
DEFAULT_SAMPLES_PER_CLASS = 3

OPTIMIZER = 'Adam'
WEIGHT_DECAY = 0

LR_SCHEDULER = 'step-lr'
SCHEDULER_STEP = 7
START_EXP_DECAY = 150
LR_DECAY = 0.1

ENCODER1 = 'encoder1'
ENCODER1_FC = 'encoder1_fc'
ENCODER2 = 'encoder2'
ENCODER4 = 'encoder4'

# this is currently ignored
# -> resnet50 is chosen 
FEATURE_EXTRACTOR = 'resnet18'

# minimal number of canny features in patches.
# 1000 seems to be a good value for binarized images
THRESHOLD_FEATURES = 1500
CANNY_SIGMA = 3

LOSSES = {'triplet'}

VAL_INTERVAL = 5

EXPERIMENT_DIR = 'experiments/'


POOL_LR_MULTIPLIER = 10
