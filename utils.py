import csv


def read_csv(csv_path):
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        lines = [line for line in csv_reader]

    return lines


def write_csv(csv_path, lines):
    with open(csv_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(lines)


def get_feature_parameters(feature):
    feature_parameters = list()

    n_mfcc_list = (12, 13, 20, 26, 40)
    n_mels_list = (64, 128, 256, 512)
    n_fft_list = (512, 1024, 2048)

    if feature == 1:
        for n_fft in n_fft_list:
            feature_parameters.append({"n_fft": n_fft})
    elif feature == 2:
        for n_mels in n_mels_list:
            for n_fft in n_fft_list:
                feature_parameters.append({"n_mels": n_mels, "n_fft": n_fft})
    else:
        for n_mfcc in n_mfcc_list:
            for n_mels in n_mels_list:
                for n_fft in n_fft_list:
                    feature_parameters.append({"n_mfcc": n_mfcc, "n_mels": n_mels, "n_fft": n_fft})
    
    return feature_parameters
