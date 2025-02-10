import streamlit as st
import pandas as pd
from policyengine.utils.charts import format_fig
from policyengine import Simulation
from policyengine.utils.huggingface import download
from pathlib import Path
import h5py
import numpy as np
import plotly.express as px
from typing import Callable

st.title("Where do I fit on the income distribution?")

st.write("This app allows you to see where you fit on the income distribution in the UK, your local constituency, and your local authority.")

@st.cache_resource
def get_simulation():
    sim = Simulation({
        "country": "uk",
        "scope": "macro",
    })
    sim.baseline_simulation.calculate("household_net_income")
    constituency_names_file_path = download(
        repo="policyengine/policyengine-uk-data",
        repo_filename="constituencies_2024.csv",
        local_folder=None,
        version=None,
    )
    constituency_names = pd.read_csv(constituency_names_file_path)
    constituency_weights_file = download(
        repo="policyengine/policyengine-uk-data",
        repo_filename="parliamentary_constituency_weights.h5",
        local_folder=None,
        version=None,
    )
    with h5py.File(constituency_weights_file, "r") as f:
        constituency_weights = f[str(2025)][...]
    return sim, constituency_names, constituency_weights

sim, constituency_names, constituency_weights = get_simulation()

def get_result(sim: Simulation, function: Callable, weights: np.ndarray):
    sim.baseline_simulation.set_input(
        "household_weight",
        sim.options.time_period,
        weights,
    )
    for variable in [
        "person_weight",
        "benunit_weight"
    ]:
        sim.baseline_simulation.get_holder(variable).delete_arrays()
    return function(sim)


def get_result_df(sim: Simulation, function: Callable, constituency_weights: np.ndarray):
    area_names = []
    results = []

    for i in range(constituency_weights.shape[0]):
        area_name = constituency_names.iloc[i]["name"]
        area_names.append(area_name)
        result = get_result(sim, function, constituency_weights[i])
        results.append(result)

    area_names.append("UK")
    results.append(get_result(sim, function, constituency_weights.sum(axis=0)))
    
    return pd.DataFrame({
        "Area": area_names,
        "Result": results
    })

def get_metric(function: Callable, target_area: str, label: str, invert_delta: bool = False, fmt: Callable = None):
    df = get_result_df(sim, function, constituency_weights).sort_values("Result").reset_index(drop=True)
    area_index = df[df["Area"] == target_area].index[0]
    area_value = df.iloc[area_index]["Result"]
    area_value_str = "(below average)" if area_index < len(df) / 2 else "(above average)"
    percentile = area_index / len(df)
    if fmt is not None:
        area_value = fmt(area_value)
    return st.metric(
        label,
        area_value,
        delta=f"Percentile {np.clip(percentile*100, 1, 99):.0f} " + area_value_str,
        delta_color=("inverse" if not invert_delta else "normal") if percentile < 0.5 else ("normal" if not invert_delta else "inverse"),
    )

target_area = st.selectbox(
    "Select the area you want to compare",
    constituency_names["name"].tolist() + ["UK"]
)

def poverty_rate(sim: Simulation):
    return round(sim.baseline_simulation.calculate("in_poverty", map_to="person").mean() * 100, 1)

def deep_poverty_rate(sim: Simulation):
    return round(sim.baseline_simulation.calculate("in_deep_poverty", map_to="person").mean() * 100, 1)

def child_poverty_rate(sim: Simulation):
    return round(sim.baseline_simulation.calculate("in_poverty", map_to="person")[sim.baseline_simulation.calculate("is_child")].mean() * 100, 1)

st.subheader("Poverty")

st.write(f"These metrics show how {target_area} compares to the rest of the UK in terms of poverty rates.")

get_metric(poverty_rate, target_area, "Poverty rate (absolute, BHC)", True)
get_metric(deep_poverty_rate, target_area, "Deep poverty rate (absolute, BHC)", True)
get_metric(child_poverty_rate, target_area, "Child poverty rate (absolute, BHC)", True)

def average_market_income(sim: Simulation):
    return round(sim.baseline_simulation.calculate("household_market_income").median())

def average_net_income(sim: Simulation):
    return round(sim.baseline_simulation.calculate("household_net_income").median())

def average_tax(sim: Simulation):
    return round(sim.baseline_simulation.calculate("household_tax").median())

def average_benefits(sim: Simulation):
    return round(sim.baseline_simulation.calculate("household_benefits").median())

st.subheader("Income")

st.write(f"These metrics show how {target_area} compares to the rest of the UK in terms of income.")

get_metric(average_market_income, target_area, "Average household market income", fmt=lambda x: f"£{x:,}")
get_metric(average_tax, target_area, "Average household tax", fmt=lambda x: f"£{x:,}")
get_metric(average_benefits, target_area, "Average household benefits", fmt=lambda x: f"£{x:,}", invert_delta=True)
get_metric(average_net_income, target_area, "Average household net income", fmt=lambda x: f"£{x:,}")

st.subheader("Income inequality")

st.write(f"These metrics show how {target_area} compares to the rest of the UK in terms of income inequality.")

def get_gini(sim: Simulation):
    return round(sim.baseline_simulation.calculate("equiv_household_net_income").gini() * 100, 1)

get_metric(get_gini, target_area, "Gini coefficient", invert_delta=True)

st.subheader("Government balances")

st.write(f"These metrics show how {target_area} compares to the rest of the UK in terms of government balances.")

def get_national_government_revenues(sim: Simulation):
    return round(sim.baseline_simulation.calculate("gov_tax").sum() - sim.baseline_simulation.calculate("council_tax").sum())

def get_national_government_expenditure(sim: Simulation):
    return round(sim.baseline_simulation.calculate("gov_spending").sum())

def get_net_national_contribution(sim: Simulation):
    return get_national_government_revenues(sim) - get_national_government_expenditure(sim)

def get_local_govt_revenues(sim: Simulation):
    return round(sim.baseline_simulation.calculate("council_tax").sum())

get_metric(get_national_government_revenues, target_area, "National government revenues", fmt=lambda x: f"£{x/1e6:,.1f} million")
get_metric(get_national_government_expenditure, target_area, "National government expenditure", fmt=lambda x: f"£{x/1e6:,.1f} million", invert_delta=True)
get_metric(get_net_national_contribution, target_area, "Net national contribution", fmt=lambda x: f"£{x/1e6:,.1f} million")
get_metric(get_local_govt_revenues, target_area, "Local government revenues", fmt=lambda x: f"£{x/1e6:,.1f} million")
