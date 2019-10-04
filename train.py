import os
from data import open_data_file
from generator import get_training_and_validation_generators
from model import build_model
from training import load_old_model, train_model

config = dict()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["labels"] = (0, 1, 2, 4)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))

config["truth_channel"] = config["nb_channels"]

config["batch_size"] = 1
config["validation_batch_size"] = 2
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 5  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 30  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 1e-4
config["optimizer"] = "Adam"
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["skip_blank"] = False  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("brats_data_isensee_2018.h5")
config["model_file"] = os.path.abspath("isensee_2018_model.h5")
config["weigths_file"] = os.path.abspath("isensee_2018_weights.h5")
config["training_file"] = os.path.abspath("isensee_training_ids.pkl")
config["validation_file"] = os.path.abspath("isensee_validation_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.

data_file_opened = open_data_file(config["data_file"])

# Tao train, validation generator
train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
    data_file_opened,
    batch_size=config["batch_size"],
    data_split=config["validation_split"],
    overwrite=config["overwrite"],
    validation_keys_file=config["validation_file"],
    training_keys_file=config["training_file"],
    n_labels=config["n_labels"],
    labels=config["labels"],
    validation_batch_size=config["validation_batch_size"],
    permute=config["permute"],
    augment=config["augment"],
    skip_blank=config["skip_blank"],
    augment_flip=config["flip"],
    augment_distortion_factor=config["distort"])


# Build model

if not config["overwrite"] and os.path.exists(config["model_file"]):
    pass
    model = load_old_model(config["model_file"])
else:
    # Tao model moi
    model = build_model(input_shape=config["input_shape"], output_channels=4, learning_rate=config["initial_learning_rate"])

# run training
train_model(model=model,
            model_file=config["model_file"],
            training_generator=train_generator,
            validation_generator=validation_generator,
            steps_per_epoch=n_train_steps,
            validation_steps=n_validation_steps,
            initial_learning_rate=config["initial_learning_rate"],
            learning_rate_drop=config["learning_rate_drop"],
            learning_rate_patience=config["patience"],
            early_stopping_patience=config["early_stop"],
            n_epochs=config["n_epochs"])

data_file_opened.close()
