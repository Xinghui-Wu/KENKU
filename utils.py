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
