import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
from dataclasses import dataclass
from typing import Any, Annotated as Ann, List, Literal as L
from pandas_dataclasses import AsDataFrame, Data, Index

"""FINAL PROJECT
AUTHOR: Albert Parlys  
PROJECT PURPOSE: This project was implemented as a requirement to complete a Python course organized by Digitalent Kominfo


"""


@dataclass
class User(AsDataFrame):
    id: Index[str]
    created_at: Data[L["datetime64[ns]"]]
    created_by: Data[str]
    updated_at: Data[L["datetime64[ns]"]]
    updated_by: Data[str]
    email: Data[str]
    is_ldap_user: Data[bool]
    status: Data[str]
    is_deleted: Data[bool]


@dataclass
class UserSession(AsDataFrame):
    id: Data[str]
    created_at: Data[L["datetime64[ns]"]]
    created_by: Data[str]
    updated_at: Data[L["datetime64[ns]"]]
    updated_by: Data[str]
    last_used: Data[L["datetime64[ns]"]]
    status: Data[str]
    type: Data[str]
    user_id: Data[str]
    android_id: Data[str]
    app_build: Data[str]
    app_name: Data[str]
    app_package: Data[str]
    app_version: Data[str]
    device_brand: Data[str]
    device_id: Data[str]
    device_manufacturer: Data[str]
    device_model: Data[str]
    device_name: Data[str]
    os_version: Data[str]
    is_deleted: Data[bool]


def extract_user_data() -> List[User]:
    """
    User csv will be read then change data types of two attributes:
    1. created_at --> datetime
    2. updated_at --> datetime
    """
    filename = "../asset/users_okay_to_publish.csv"
    with open(filename) as users_csv:
        user_df = pd.read_csv(users_csv)
        user_df["created_at"] = pd.to_datetime(user_df["created_at"])
        user_df["updated_at"] = pd.to_datetime(user_df["updated_at"])
        users_obj = [User(**kwargs) for kwargs in user_df.to_dict(orient="records")]
        return users_obj


def extract_session_data() -> List[UserSession]:
    """
    Session csv will be read then change data types of two attributes:
    1. created_at --> datetime
    2. updated_at --> datetime
    3. last_used --> datetime
    """
    filename = "../asset/sessions_okay_to_publish.csv"
    with open(filename) as users_csv:
        session_df = pd.read_csv(users_csv)
        session_df["created_at"] = pd.to_datetime(session_df["created_at"])
        session_df["updated_at"] = pd.to_datetime(session_df["updated_at"])
        session_df["last_used"] = pd.to_datetime(session_df["last_used"])
        sessions_obj = [
            UserSession(**kwargs) for kwargs in session_df.to_dict(orient="records")
        ]
        return sessions_obj


def data_cleaning(session_data: pd.DataFrame, user_data: pd.DataFrame) -> pd.DataFrame:
    """Data analysis requirement:
    1. Session:
        - only "VALID" status
        - only android device
        - only session on this year
    2. User:
        - only user with android session available

    Args:
        session_data (pd.DataFrame): extracted dataframe from csv
        user_data (pd.DataFrame): extracted dataframe from csv

    Returns:
        pd.DataFrame: cleaned dataframe from function
    """
    today_datetime = datetime.datetime.today()
    today_datetime.strftime("%Y%m%d")
    filtered_datetime = datetime.datetime(day=1, month=1, year=today_datetime.year)

    print(
        f"program will filter session data from {today_datetime.strftime('%Y/%m/%d')} to {filtered_datetime.strftime('%Y/%m/%d')}\n"
    )

    filtered_session_data_step_2 = session_data.query(
        f'last_used > {filtered_datetime.strftime("%Y%m%d")}'
    )

    print(f"data shape comparisson: \n\tbefore: ")
    print(f"\t\tsession_df shape: {session_data.shape}")
    print(f"\tafter: ")
    print(f"\t\tfiltered_sesion_df shape: {filtered_session_data_step_2.shape}")

    print(f"program will filter session data to VALID only\n")

    filtered_session_data_step_3 = filtered_session_data_step_2.drop(
        filtered_session_data_step_2[
            filtered_session_data_step_2["status"] != "VALID"
        ].index
    )
    print(f"data shape comparisson: \n\tbefore: ")
    print(
        f"\t\tfiltered_session_data_step_2 shape: {filtered_session_data_step_2.shape}"
    )
    print(f"\tafter: ")
    print(
        f"\t\tfiltered_session_data_step_3 shape: {filtered_session_data_step_3.shape}"
    )

    print(f"program will filter session data to Android devices only\n")

    filtered_session_data_step_4 = filtered_session_data_step_3.dropna(
        subset=["os_version"]
    )
    filtered_session_data_step_4 = filtered_session_data_step_4.drop(
        filtered_session_data_step_4[
            filtered_session_data_step_4["device_model"] == "iPhone"
        ].index
    )
    filtered_session_data_step_4 = filtered_session_data_step_4.drop(
        filtered_session_data_step_4[
            filtered_session_data_step_4["device_model"] == "iPad"
        ].index
    )
    print(f"data shape comparisson: \n\tbefore: ")
    print(
        f"\t\tfiltered_session_data_step_3 shape: {filtered_session_data_step_3.shape}"
    )
    print(f"\tafter: ")
    print(
        f"\t\tfiltered_session_data_step_4 shape: {filtered_session_data_step_4.shape}"
    )

    print(f"program will join table user data and filtered sesion data\n")

    filtered_session_data_step_5 = user_data.join(
        filtered_session_data_step_4.set_index("user_id"), on="id", rsuffix="_session"
    )
    print(f"data shape comparisson: \n\tbefore: ")
    print(f"\t\tuser_data shape: {user_data.shape}")
    print(
        f"\t\tfiltered_session_data_step_4 shape: {filtered_session_data_step_4.shape}"
    )
    print(f"\tafter: ")
    print(
        f"\t\tfiltered_session_data_step_5 shape: {filtered_session_data_step_5.shape}"
    )

    print(f"program will filter data to user with android device only\n")

    filtered_session_data_step_6 = filtered_session_data_step_5.drop_duplicates(
        subset=["id"]
    )
    filtered_session_data_step_6 = filtered_session_data_step_6.dropna(
        subset="os_version"
    )

    # Check no null data
    check_null_1 = filtered_session_data_step_6[["os_version"]].isnull().sum()
    print(f"check no null data on os_version")
    print(f"\t{check_null_1}\n")

    check_null_2 = filtered_session_data_step_6[["app_version"]].isnull().sum()
    print(f"check no null data on app_version")
    print(f"\t{check_null_2}\n")

    print(f"data shape comparisson: \n\tbefore: ")
    print(
        f"\t\tfiltered_session_data_step_5 shape: {filtered_session_data_step_5.shape}"
    )
    print(f"\tafter: ")
    print(
        f"\t\tfiltered_session_data_step_6 shape: {filtered_session_data_step_6.shape}"
    )
    data_checkpoint_cleaned = filtered_session_data_step_6
    return data_checkpoint_cleaned


def data_analysis(data_checkpoint_cleaned: pd.DataFrame) -> None:
    """This function does data analysis for user and session data

    Args:
        data_checkpoint_cleaned (pd.DataFrame): data cleaned from previous step
    """

    def plot1(data_checkpoint_cleaned: pd.DataFrame) -> None:
        """Graph that shows os_version distribution on users

        Args:
            data_checkpoint_cleaned (pd.DataFrame): data cleaned from previous step
        """
        android_by_os_version = (
            data_checkpoint_cleaned.groupby(["os_version"])["os_version"]
            .count()
            .reset_index(name="jumlah")
        )
        android_by_os_version = android_by_os_version.astype({"os_version": int})
        # plot configuration
        font = {
            "family": "serif",
            "color": "black",
            "weight": "bold",
            "size": 24,
        }
        colors = [
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "lime",
            "lime",
            "lime",
            "lime",
            "lime",
        ]
        plt.style.use("_mpl-gallery")
        x = android_by_os_version.os_version
        y = android_by_os_version.jumlah

        fig, ax = plt.subplots(
            figsize=(16, 12),
        )

        p1 = ax.bar(
            x, y, width=1, edgecolor="white", linewidth=0.7, label=y, color=colors
        )
        ax.set(xlim=(x.min() - 1, x.max() + 1), xticks=np.arange(x.min(), x.max() + 1))
        ax.bar_label(p1)
        plt.xlabel("OS Version")
        plt.ylabel("Jumlah")
        plt.title("Data Sebaran OS Version pada Pengguna Aplikasi", fontdict=font)

        # save fig
        plt.savefig("../target/graph_1_osversion.png", dpi=300, bbox_inches="tight")

        # plt.show()

        user_under_expected_os_version = (
            android_by_os_version[android_by_os_version["os_version"] < 29].sum().jumlah
        )
        total_users = android_by_os_version["jumlah"].sum()
        print(
            f"total user under os_version: 29 is \n{user_under_expected_os_version} ({(user_under_expected_os_version/total_users)*100:.2f}%)\n(val,(pct))"
        )

    def plot2(data_checkpoint_cleaned: pd.DataFrame) -> None:
        """Graph that shows app_version distribution on users

        Args:
            data_checkpoint_cleaned (pd.DataFrame): data cleaned from previous step
        """
        android_by_app_version = (
            data_checkpoint_cleaned.groupby(["app_version"])["app_version"]
            .count()
            .reset_index(name="jumlah")
        )

        android_by_app_version = android_by_app_version.astype({"app_version": int})
        android_by_app_version = android_by_app_version.astype({"app_version": str})

        # plot configuration
        font = {
            "family": "serif",
            "color": "black",
            "weight": "bold",
            "size": 24,
        }
        colors = [
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "tomato",
            "lime",
            "lime",
        ]

        plt.style.use("_mpl-gallery")
        x = android_by_app_version.app_version
        y = android_by_app_version.jumlah

        fig, ax = plt.subplots(
            figsize=(16, 12),
        )
        width = 1
        p1 = ax.bar(
            x, y, width=width, edgecolor="white", linewidth=0.7, label=y, color=colors
        )
        ax.bar_label(p1)
        plt.xlabel("APP Version")
        plt.ylabel("Jumlah")
        plt.title("Data Sebaran APP Version pada Pengguna Aplikasi", fontdict=font)

        # save fig
        plt.savefig("../target/graph_2_appversion", dpi=300, bbox_inches="tight")

        # plt.show()

        android_by_app_version = android_by_app_version.astype({"app_version": int})

        user_under_expected_app_version = (
            android_by_app_version[android_by_app_version["app_version"] < 200]
            .sum()
            .jumlah
        )
        total_users = android_by_app_version["jumlah"].sum()
        print(
            f"total user under app_version: 200 is \n{user_under_expected_app_version} ({(user_under_expected_app_version/total_users)*100:.2f}%)\n(val,(pct))"
        )

    def plot3(data_checkpoint_cleaned: pd.DataFrame) -> None:
        """Graph that shows os_version x app_version distribution on users

        Args:
            data_checkpoint_cleaned (pd.DataFrame): data cleaned from previous step
        """
        android_by_version_and_app_version = (
            data_checkpoint_cleaned.groupby(["os_version", "app_version"])["os_version"]
            .count()
            .reset_index(name="jumlah")
        )
        android_by_version_and_app_version = android_by_version_and_app_version.astype(
            {"app_version": int, "os_version": int}
        )
        android_by_version_and_app_version = android_by_version_and_app_version.astype(
            {"app_version": str, "os_version": str}
        )

        data = android_by_version_and_app_version
        labels_unique = data.app_version.unique()
        labels_unique = labels_unique.astype(int)
        labels_unique = np.msort(labels_unique)

        # plot
        hue_order = [
            "104",
            "105",
            "106",
            "107",
            "108",
            "109",
            "120",
            "130",
            "131",
            "200",
            "201",
        ]
        g = sns.displot(
            data=data,
            x="os_version",
            hue="app_version",
            weights="jumlah",
            discrete=True,
            multiple="stack",
            height=10,
            hue_order=hue_order,
        )
        min_height = 10

        # iterate through each axes
        for ax in g.axes.flat:

            # iterate through each container
            for c in ax.containers:

                # Optional: if the segment is small or 0, customize the labels
                labels = [
                    v.get_height() if v.get_height() > min_height else "" for v in c
                ]

                # remove the labels parameter if it's not needed for customized labels
                ax.bar_label(c, labels=labels, label_type="center")
        ax.set_ylabel("Jumlah")
        ax.set_xlabel("OS Version")
        ax.set_title(
            "Data Sebaran APP Version terhadap OS Version pada Pengguna Aplikasi"
        )

        # save fig
        plt.savefig(
            "../target/graph_3_appversion_vs_os_version", dpi=300, bbox_inches="tight"
        )

        # plt.show()

        # user_under_expected refer to user_under_expected_app_version_by_compatible_os_version
        android_by_version_and_app_version = android_by_version_and_app_version.astype(
            {"app_version": int, "os_version": int}
        )
        user_under_expected = (
            android_by_version_and_app_version[
                (android_by_version_and_app_version["app_version"] < 200)
                & (android_by_version_and_app_version["os_version"] >= 29)
            ]
            .sum()
            .jumlah
        )
        total_users = android_by_version_and_app_version["jumlah"].sum()
        print(
            f"total user under app_version: 200 but compatible above os_version: 29 is \n{user_under_expected} ({(user_under_expected/total_users)*100:.2f}%)\n(val,(pct))"
        )

    plot1(data_checkpoint_cleaned=data_checkpoint_cleaned)
    plot2(data_checkpoint_cleaned=data_checkpoint_cleaned)
    plot3(data_checkpoint_cleaned=data_checkpoint_cleaned)


if __name__ == "__main__":
    print("Starting Program from executables")
    print("-------------------------------------------------------------")
    session_data = pd.DataFrame(extract_session_data())
    user_data = pd.DataFrame(extract_user_data())
    data_checkpoint_cleaned = data_cleaning(
        session_data=session_data, user_data=user_data
    )
    data_analysis(data_checkpoint_cleaned=data_checkpoint_cleaned)
    print("-------------------------------------------------------------")
    print("Program Successfuly running")
    print("You can find your plot here:")
    print(
        f"Plot 1: {os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/target/graph_1_osversion.png"
    )
    print(
        f"Plot 2: {os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/target/graph_2_appversion.png"
    )
    print(
        f"Plot 3: {os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/target/graph_3_appversion_vs_os_version.png"
    )
    print("-------------------------------------------------------------")
