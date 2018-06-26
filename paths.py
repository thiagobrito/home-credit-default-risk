import os


def current_directory():
    return os.path.abspath(os.path.dirname(__file__))


def results_directory():
    results = os.path.join(current_directory(), 'results')
    if not os.path.exists(results):
        os.makedirs(results)
    return results


def make_results_path(model_name, type, scores):
    results_path = os.path.join(results_directory(), '%s_%s_%f' % (model_name, type, scores))
    return results_path.replace('.', '_') + '.csv'


def indices_path():
    return os.path.join(results_directory(), 'indices.json')
