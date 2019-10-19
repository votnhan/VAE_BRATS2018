import tables
import os
import numpy as np
import nibabel as nib
from utils import pickle_load
from training import load_old_model


def run_validation_cases(validation_keys_file, model_file, training_modalities, labels, hdf5_file,
                         output_label_map=False, output_dir="."):
    
    validation_indices = pickle_load(validation_keys_file)
    model = load_old_model(model_file)
    data_file = tables.open_file(hdf5_file, "r")
    for index in validation_indices:
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))
        run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            training_modalities=training_modalities, output_label_map=output_label_map, labels=labels)
    
    data_file.close()


def run_validation_case(data_index, output_dir, model, data_file, training_modalities,
                        output_label_map=False, labels=None, permute=False):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))

    test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
    test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))

    prediction = predict(model, test_data, permute=permute)
    prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, labels=labels)

    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))


def predict(model, data, permute=False):
    if permute:
        # Chua can dung
        pass
    else:
        return model.predict(data)


def prediction_to_image(prediction, affine, label_map=False, labels=None):
    
    if label_map:
        label_map_data = get_prediction_labels(prediction, labels=labels)
        data = label_map_data[0]
    else:
        return multi_class_prediction(prediction, affine)

    return nib.Nifti1Image(data, affine)

def get_prediction_labels(prediction, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        if labels:
            for value in np.unique(label_data).tolist():
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays

def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images