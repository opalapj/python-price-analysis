from datetime import date
from datetime import datetime
from pathlib import Path

import holidays as hd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MarketData:
    def __init__(self):
        print("Start of object initiation.")
        data = pd.read_csv(
            filepath_or_buffer="data/input/prices.csv",
            index_col=["date [-]"],
            parse_dates=True,
        )
        data.index = pd.DatetimeIndex(data.index, tz="Europe/Warsaw")
        variables_list = data.columns.to_list()
        self.variables_dict = dict(enumerate(variables_list))
        print("Available variables collected.")
        data.insert(0, "date", data.index.date)
        data.insert(0, "year", data.index.year)
        data.insert(0, "quarter", data.index.quarter)
        data.insert(0, "month", data.index.month)
        data.insert(0, "week", data.index.isocalendar()["week"])
        data.insert(0, "monthday", data.index.day)
        data.insert(0, "weekday", data.index.weekday + 1)
        hdays = hd.country_holidays("PL", years=data.index.year.unique())
        holidays_mask = np.where(
            data["date"].isin(hdays) | data["weekday"].isin([6, 7]), 0, 1
        )
        data.insert(0, "workday", holidays_mask)
        data.insert(0, "hour", data.index.hour + 1)
        midday_hour_start, midday_hour_stop = 10, 13
        midday_mask = np.where(
            (data["hour"] >= midday_hour_start) & (data["hour"] <= midday_hour_stop),
            1,
            0,
        )
        data.insert(0, "midday", midday_mask)
        print("Additional columns added (year, quarter, month, etc.).")
        self.date_range_dict = {
            "start date": data.index.min(),
            "end date": data.index.max(),
        }
        print("Available date range collected.")
        print("End of object initiation.")
        self.holidays = hdays
        self.data = data
        self.matrices_dict = None
        self.data_filtered = None
        self.ratios_dict = None

    def matrices(self, workday=None, weekday=None):
        data_filtered = self.data.copy()
        print("Available variables:")
        dictionary_listing(self.variables_dict)
        print("Add variables to matrices creating.")
        # vars_selected = stack(self.variables_dict)
        # vars_selected = [0, 1, 2, 3]  # Input for test.
        vars_selected = [0, 1, 2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        print("Available range of data:")
        dictionary_listing(self.date_range_dict)
        print("Available date range modes:")
        modes_dict = {0: "full date range", 1: "selected date range"}
        dictionary_listing(modes_dict)
        # mode = modes_dict[get_key_from_dictionary(modes_dict)]
        mode = "full date range"  # Input for test.
        if mode == "selected date range":
            start_date = date_enter_and_validate(strg="start date")
            end_date = date_enter_and_validate(strg="end date")
            data_filtered = data_filtered.loc[
                data_filtered["date"].between(start_date, end_date)
            ]
        if workday is not None:
            data_filtered = data_filtered.loc[data_filtered["workday"] == workday]
        if weekday is not None:
            data_filtered = data_filtered.loc[data_filtered["weekday"] == weekday + 1]
        if not data_filtered.empty:
            matrices_list = []
            for var in vars_selected:
                var_ = self.variables_dict[var]
                m = data_filtered.pivot_table(
                    index=["year", "month"], columns="hour", values=var_
                )
                matrices_list.append((var_, m))
            self.matrices_dict = {k: v for k, v in enumerate(matrices_list)}
        self.data_filtered = data_filtered

    def standard_deviation(self):
        print("Available variables:")
        dictionary_listing(self.variables_dict)
        print("Add variables to standard deviation calculation.")
        # vars_selected = stack(self.variables_dict)
        vars_selected = [0, 1]  # Input for test.
        for var in vars_selected:
            var_ = self.variables_dict[var]
            m = self.data_filtered.pivot_table(
                index=["year", "month"], columns="hour", values=var_, aggfunc="std"
            )
            self.matrices_dict[max(self.matrices_dict.keys()) + 1] = (
                var_[: var_.find("[")] + "std " + var_[var_.find("[") :],
                m,
            )

    def forecast_error(self):
        print("Available matrices:")
        dictionary_listing(self.matrices_dict, value_type="sequence")
        print("Choose 1st matrix. ", end="")
        # m_to_error_1 = self.matrices_dict[
        #     get_key_from_dictionary(self.matrices_dict)
        # ][1]
        m_to_error_1 = self.matrices_dict[2][1]  # Input for test.
        print("Choose 2nd matrix. ", end="")
        # m_to_error_2 = self.matrices_dict[
        #     get_key_from_dictionary(self.matrices_dict)
        # ][1]
        m_to_error_2 = self.matrices_dict[3][1]  # Input for test.
        m_error = (m_to_error_1 - m_to_error_2) / m_to_error_1
        self.matrices_dict[max(self.matrices_dict.keys()) + 1] = (
            "load diff [%]",
            m_error,
        )

    def normalization(self):
        print("Available matrices:")
        dictionary_listing(self.matrices_dict, value_type="sequence")
        print("Add matrices to normalize.")
        # m_to_norm = stack(self.matrices_dict)
        m_to_norm = [0, 1]  # Input for test.
        for m in m_to_norm:
            m_ = self.matrices_dict[m]
            m_df = m_[1]
            m_norm = m_df.div(m_df.mean(axis=1), axis=0)
            self.matrices_dict[max(self.matrices_dict.keys()) + 1] = (
                m_[0][: m_[0].find("[")] + "normalized [-]",
                m_norm,
            )

    def heatmaps(self):
        print("Start of heatmaps creation.")
        print("Available matrices:")
        dictionary_listing(self.matrices_dict, value_type="sequence")
        print("Add matrices to heatmap creation.")
        # m_to_heatmaps = stack(self.matrices_dict)
        # m_to_heatmaps = [0, 2, 4]  # Input for test.
        m_to_heatmaps = list(range(len(self.matrices_dict)))
        for m in m_to_heatmaps:
            m_ = self.matrices_dict[m]
            draw_heatmap(m_)
        print("End of heatmaps creation.")

    def correlation(self):
        data_corr = self.data_filtered.iloc[:, 10:]
        data_corr = data_corr.corr(numeric_only=True)
        fig, ax = plt.subplots(
            figsize=(data_corr.shape[0] * 0.5, data_corr.shape[0] * 0.5)
        )
        mask = np.triu(np.ones_like(data_corr, dtype=bool), k=1)
        sns.heatmap(
            data_corr,
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            vmin=-1.0,
            vmax=1.0,
            center=0,
            square=True,
            annot=True,
            mask=mask,
            cbar=False,
            ax=ax,
            fmt=".2f",
        )
        plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
        plt.tight_layout()
        # plt.show()
        file_path = Path("data", "output", "correlation")
        plt.savefig(file_path)
        plt.close(fig)

    def ratios(self):
        data_ratios = self.data.loc[
            ~((self.data["month"] == 1) & (self.data["week"] > 5))
        ]
        data_ratios = data_ratios.loc[data_ratios["workday"] == 1]
        data_ratios_gb = data_ratios.groupby(["year", "week", "date"])[
            ["midday", "day ahead price [PLN/MWh]", "imbalance price [PLN/MWh]"]
        ]

        def ratios_daily_calc(g):
            g.set_index("midday", inplace=True)
            mean_midday = g.loc[1].mean()
            max_rest = g.loc[0].max()
            return mean_midday / max_rest

        ratios_daily = data_ratios_gb.apply(ratios_daily_calc)
        ratios_daily.reset_index(level=-1, drop=True, inplace=True)
        ratios_weekly_gb = ratios_daily.groupby(level=ratios_daily.index.names)
        ratios_weekly = ratios_weekly_gb.mean()
        ratios_list = []
        for c in ratios_weekly.columns:
            name = c[: c.find("[")] + "ratio [-]"
            s = ratios_weekly[c].rename(name)
            df = s.unstack(0)
            ratios_list.append((name, df))
        self.ratios_dict = {k: v for k, v in enumerate(ratios_list)}
        fig, axs = plt.subplots(
            len(self.ratios_dict),
            1,
            figsize=(15, 10),
            layout="constrained",
            sharex=True,
        )
        for k, v in self.ratios_dict.items():
            axs[k].plot(v[1], label=v[1].columns)
            axs[k].grid(visible=True)
            axs[k].set_title(v[0])
        else:
            axs[k].set(xlabel="week")
            handles, labels = axs[k].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
        # plt.tight_layout()
        # plt.show()
        file_path = Path("data", "output", "ratios")
        plt.savefig(file_path)


def date_enter_and_validate(strg="date", typ="date"):
    text = "Enter " + strg + " in format yyyy-mm-dd: "
    dat = input(text)
    try:
        valid_date = datetime.strptime(dat, "%Y-%m-%d")
        if typ == "date":
            return valid_date.date()
        else:
            return valid_date
    except ValueError:
        print("Invalid date!")
        return date_enter_and_validate(strg)


def csv_reading(path, sep=";", decimal=",", na_values="#DZIEL/0!", encoding="utf-8"):
    try:
        csv_data = pd.read_csv(
            path, sep=sep, decimal=decimal, na_values=na_values, encoding=encoding
        )
        print("Data read correctly.")
        return csv_data
    except FileNotFoundError:
        print("Invalid file path!")
    except UnicodeDecodeError:
        encoding = "1250"
        print(f"Encoding changed from utf-8 to {encoding}")
        return csv_reading(path, encoding=encoding)


def get_key_from_dictionary(dictionary):
    try:
        key = int(input("Enter index: "))
        if key in dictionary.keys():
            return key
        else:
            raise KeyError
    except (KeyError, ValueError):
        print("Invalid index!")
        return get_key_from_dictionary(dictionary)


def stack(dictionary):
    s = []
    while True:
        s.append(get_key_from_dictionary(dictionary))
        inpt = None
        while inpt not in ["b", "c"]:
            inpt = input("Enter 'b' to break adding or 'c' to continue: ")
        if inpt == "b":
            break
        elif inpt == "c":
            continue
    return s


def dictionary_listing(dictionary, value_type=None, value_index=0):
    if value_type == "sequence":
        for key, value in dictionary.items():
            print(key, "->", value[value_index])
    else:
        for key, value in dictionary.items():
            print(key, "->", value)


def draw_heatmap(matrix):
    m_name, m_df = matrix[0], matrix[1]
    m_hours = m_df.columns.to_list()
    m_periods = [date(a, b, 1) for a, b in m_df.index.to_list()]
    hours_mg, period_mg = np.meshgrid(m_hours, m_periods)
    var_str = m_name[: (m_name.find("[") - 1)].replace(" ", "_")
    fig, ax = plt.subplots(figsize=(len(m_hours) * 0.75, len(m_periods) * 0.35))
    ax.pcolormesh(
        hours_mg,
        period_mg,
        m_df,
        cmap="jet",
        edgecolors="grey",
        linewidth=0.5,
        alpha=0.5,
    )
    ax.set_title(m_name)
    ax.xaxis.set_ticks(m_hours)
    sec_xax = ax.secondary_xaxis("top")
    sec_xax.xaxis.set_ticks(m_hours)
    ax.yaxis.set_ticks(m_periods)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    sec_yax = ax.secondary_yaxis("right")
    sec_yax.yaxis.set_ticks(m_periods)
    sec_yax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    unit = m_name[(m_name.find("[")) + 1 : -1]
    if unit == "%":
        m_df = m_df.map(lambda v: "{:.0%}".format(v))
    elif unit == "-":
        m_df = m_df.map(lambda v: "{:.2f}".format(v))
    else:
        m_df = m_df.map(lambda v: "{:.0f}".format(v))
    for j in range(m_df.shape[0]):
        for i in range(m_df.shape[1]):
            plt.text(
                m_hours[i], m_periods[j], m_df.iloc[j, i], ha="center", va="center"
            )
    plt.tight_layout()
    # plt.show()
    file_name = "_".join(
        [var_str, m_periods[0].strftime("%Y-%m"), m_periods[-1].strftime("%Y-%m")]
    )
    file_path = Path("data", "output", file_name)
    plt.savefig(file_path)
    plt.close(fig)


kse = MarketData()
for wd in range(2, 3):
    kse.matrices(workday=1, weekday=wd)
    kse.standard_deviation()
    kse.forecast_error()
    kse.normalization()
    kse.heatmaps()
    kse.correlation()
    kse.ratios()
