from os import path


def data_path(filename):
    data_dir = path.join(path.dirname(__file__), 'data')
    return path.join(data_dir, filename)
