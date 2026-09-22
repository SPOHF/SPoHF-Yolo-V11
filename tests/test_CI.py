import os

# repo root, so the tests don't care where pytest is started from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(*parts):
    return os.path.join(ROOT, *parts)


# Check weights
def test_weights_files_exist():
    for name in ["best.pt", "last.pt"]:
        p = path("Best-Model-Weights", name)
        assert os.path.isfile(p), f"Best-Model-Weights/{name} does not exist"


# Weights should be real files, not LFS pointers / empty placeholders
def test_weights_not_empty():
    for name in ["best.pt", "last.pt"]:
        p = path("Best-Model-Weights", name)
        assert os.path.getsize(p) > 1_000_000, f"Best-Model-Weights/{name} looks truncated"


# Check structure
def test_required_folders_exist():
    required = [
        "Best-Model-Weights",
        "train",
        "test",
        "valid",
        "tests",
        "runs",
    ]
    for folder in required:
        assert os.path.isdir(path(folder)), f"{folder} is missing"


# Check the scripts that are supposed to be runnable
def test_required_scripts_exist():
    required = [
        "SPoHF-predict.py",
        "SPoHF-predict-save-insects.py",
        "trainTheModel.py",
        "historical_data_analysis.py",
        "twin_CSV_export.py",
    ]
    for script in required:
        assert os.path.isfile(path(script)), f"{script} is missing"


# check requirements.txt
def test_requirements_exists():
    assert os.path.isfile(path("requirements.txt"))


# dataset config the training script needs
def test_data_yaml_exists():
    assert os.path.isfile(path("data.yaml"))


# .env is gitignored, so the example has to be there for anyone cloning
def test_env_example_exists():
    assert os.path.isfile(path(".env-examle")), ".env example file is missing"
