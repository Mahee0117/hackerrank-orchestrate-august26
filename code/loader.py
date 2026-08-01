from pathlib import Path
import pandas as pd


# Get the project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folder
DATASET_PATH = PROJECT_ROOT / "dataset"


def load_data():
    data = {
        "messages": pd.read_csv(DATASET_PATH / "messages.csv"),
        "users": pd.read_csv(DATASET_PATH / "users.csv"),
        "message_history": pd.read_csv(DATASET_PATH / "message_history.csv"),
        "message_events": pd.read_csv(DATASET_PATH / "message_events.csv"),
        "groups": pd.read_csv(DATASET_PATH / "groups.csv"),
        "group_members": pd.read_csv(DATASET_PATH / "group_members.csv"),
        "business_accounts": pd.read_csv(DATASET_PATH / "business_accounts.csv"),
        "user_business_history": pd.read_csv(DATASET_PATH / "user_business_history.csv"),
    }

    return data