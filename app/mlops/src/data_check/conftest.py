import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_age", action="store")
    parser.addoption("--max_age", action="store")
    parser.addoption("--min_tenure", action="store")
    parser.addoption("--max_tenure", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path, index_col='Client')

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path, index_col='Client')

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_age(request):
    min_age = request.config.option.min_age

    if min_age is None:
        pytest.fail("You must provide min_age")

    return int(min_age)

@pytest.fixture(scope='session')
def max_age(request):
    max_age = request.config.option.max_age

    if max_age is None:
        pytest.fail("You must provide max_age")

    return int(max_age)

@pytest.fixture(scope='session')
def min_tenure(request):
    min_tenure = request.config.option.min_tenure

    if min_tenure is None:
        pytest.fail("You must provide min_tenure")

    return int(min_tenure)

@pytest.fixture(scope='session')
def max_tenure(request):
    max_tenure = request.config.option.max_tenure

    if max_tenure is None:
        pytest.fail("You must provide max_tenure")

    return int(max_tenure)
