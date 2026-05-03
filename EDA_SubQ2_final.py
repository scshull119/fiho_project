# %% [markdown]
# # Sub-Question 2: Home Equity vs. Financial Assets in Wealth Accumulation
# **How important is home equity in explaining and predicting total household wealth,
# and how does its contribution compare to renters' financial assets at similar income levels?**
#
# Data: Survey of Consumer Finances (SCF) — cleaned dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.color":        "#e5e5e5",
    "grid.linewidth":    0.6,
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "axes.labelcolor":   "#333333",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "legend.frameon":    False,
    "legend.fontsize":   9,
})

BLUE  = "#2166ac"
RED   = "#d6604d"
LBLUE = "#92c5de"
LRED  = "#f4a582"
GRAY  = "#888888"

# %%
df_raw = pd.read_excel("df_model.xlsx")
df     = df_raw[df_raw["implicate"] == 1].copy()
print(f"Dataset shape (implicate 1): {df.shape}")

# %%
df["home_equity"]       = df["home_value"] - df["mortgage_balance"]
df["financial_assets"]  = (df["checking_balance_1"] + df["checking_balance_other"] +
                            df["stocks_value"] + df["bonds_value"] +
                            df["mutual_funds_value"] + df["money_market_value"])
df["total_liabilities"] = (df["mortgage_balance"] + df["credit_card_balance"] +
                            df["vehicle_debt_total"] + df["heloc_balance"])
df["total_wealth"]      = (df["home_value"] + df["financial_assets"] +
                            df["vehicle_value_total"] + df["business_value"] +
                            df["other_real_estate_value"]) - df["total_liabilities"]

df["is_homeowner"] = (df["home_value"] > 0).astype(int)
df["tenure"]       = df["is_homeowner"].map({1: "Homeowner", 0: "Renter"})

owners  = df[df["is_homeowner"] == 1].copy()
renters = df[df["is_homeowner"] == 0].copy()

print(f"Homeowners: {len(owners):,}  |  Renters: {len(renters):,}")

# %% [markdown]
# ## 1 . Summary Statistics

# %%
summary_cols = ["total_wealth", "home_equity", "financial_assets", "total_income"]
labels       = ["Total Wealth", "Home Equity", "Financial Assets", "Total Income"]

rows = []
for grp, name in [(owners, "Homeowner"), (renters, "Renter")]:
    for col, label in zip(summary_cols, labels):
        rows.append({"Tenure": name, "Variable": label,
                     "Mean": grp[col].mean(), "Median": grp[col].median(),
                     "Std Dev": grp[col].std()})

summary_df = pd.DataFrame(rows)

def fmt(x):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    elif abs(x) >= 1_000:
        return f"${x/1_000:.1f}K"
    else:
        return f"${x:.0f}"

for col in ["Mean", "Median", "Std Dev"]:
    summary_df[col] = summary_df[col].apply(fmt)

print(summary_df.to_string(index=False))

# %% [markdown]
# ## 2 . Median Wealth Comparison: Homeowners vs. Renters

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))

groups = ["Homeowners", "Renters"]
vals   = [owners["total_wealth"].median(), renters["total_wealth"].median()]
colors = [BLUE, RED]
bars   = ax.bar(groups, vals, color=colors, width=0.45, zorder=3)

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.015,
            f"${val:,.0f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=bar.get_facecolor())

ax.set_title("Median Total Wealth by Housing Tenure")
ax.set_ylabel("Median Total Wealth (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_ylim(0, max(vals) * 1.2)
ax.tick_params(axis="x", length=0)
fig.text(0.5, -0.02,
         "Figure 1: Median total wealth for homeowners and renters in the SCF sample.",
         ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
plt.savefig("fig1_median_wealth.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3 . Asset Composition by Tenure

# %%
owner_equity = owners["home_equity"].median()
owner_fin    = owners["financial_assets"].median()
renter_fin   = renters["financial_assets"].median()
renter_other = (renters["vehicle_value_total"] + renters["business_value"] +
                renters["other_real_estate_value"]).median()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
pie_kw = dict(startangle=140,
              wedgeprops=dict(edgecolor="white", linewidth=1.5),
              textprops=dict(fontsize=9))

axes[0].pie([owner_equity, owner_fin],
            labels=["Home Equity", "Financial Assets"],
            colors=[BLUE, LBLUE], autopct="%1.1f%%", **pie_kw)
axes[0].set_title("Homeowner Asset Mix\n(Medians)")

axes[1].pie([renter_fin, max(renter_other, 1)],
            labels=["Financial Assets", "Other Assets"],
            colors=[RED, LRED], autopct="%1.1f%%", **pie_kw)
axes[1].set_title("Renter Asset Mix\n(Medians)")

fig.text(0.5, -0.02,
         "Figure 2: Median asset composition for homeowners and renters. "
         "Home equity dominates homeowner wealth.",
         ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
plt.savefig("fig2_asset_composition.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4 . Wealth Distribution by Tenure

# %%
cap     = 2_000_000
df_plot = df[df["total_wealth"].between(0, cap)].copy()

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df_plot, x="tenure", y="total_wealth", hue="tenure",
            palette={"Homeowner": BLUE, "Renter": RED},
            order=["Homeowner", "Renter"], width=0.35, linewidth=0.9,
            flierprops=dict(marker="o", markersize=2.5, alpha=0.3,
                            markeredgewidth=0, markerfacecolor=GRAY),
            ax=ax, legend=False)

ax.set_title("Distribution of Total Wealth by Housing Tenure")
ax.set_xlabel("")
ax.set_ylabel("Total Wealth (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.tick_params(axis="x", length=0)
ax.annotate("Capped at $2M for readability", xy=(0.99, 0.97),
            xycoords="axes fraction", ha="right", va="top",
            fontsize=7.5, color=GRAY, style="italic")
fig.text(0.5, -0.02,
         "Figure 3: Box plots of total wealth by tenure. "
         "The median homeowner wealth substantially exceeds renter wealth.",
         ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
plt.savefig("fig3_wealth_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5 . Home Equity vs. Total Wealth (Homeowners)

# %%
# Fit on capped data so the line matches the visible points
cap_x = owners["home_equity"].quantile(0.99)
cap_y = owners["total_wealth"].quantile(0.99)
owners_capped = owners[(owners["home_equity"] <= cap_x) & (owners["total_wealth"] <= cap_y)]

m, b, r, p, _ = stats.linregress(owners_capped["home_equity"], owners_capped["total_wealth"])
r_full = owners["home_equity"].corr(owners["total_wealth"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(owners_capped["home_equity"], owners_capped["total_wealth"],
           alpha=0.25, color=BLUE, edgecolors="none", s=22, zorder=2)

ax.plot([0, cap_x], [b, m * cap_x + b], color="navy",
        linewidth=1.8, label=f"OLS fit  (r = {r:.2f})", zorder=3)

ax.set_title("Home Equity vs. Total Wealth - Homeowners")
ax.set_xlabel("Home Equity (USD)")
ax.set_ylabel("Total Wealth (USD)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.set_xlim(0, cap_x)
ax.set_ylim(0, cap_y)
ax.legend(loc="upper left")
ax.annotate("Capped at 99th percentile", xy=(0.99, 0.03),
            xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7.5, color=GRAY, style="italic")
fig.text(0.5, -0.02,
         f"Figure 4: Scatter plot of home equity vs. total wealth for homeowners (n = {len(owners_capped):,}, capped). "
         f"Pearson r = {r:.2f}.",
         ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
plt.savefig("fig4_equity_vs_wealth.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Pearson r on capped data (owners): {r:.4f}")
print(f"Pearson r on full data   (owners): {r_full:.4f}")

# %% [markdown]
# ## 6 . Financial Assets vs. Total Wealth (Renters)

# %%
cap_x2 = renters["financial_assets"].quantile(0.99)
cap_y2 = renters["total_wealth"].quantile(0.99)
renters_capped = renters[(renters["financial_assets"] <= cap_x2) & (renters["total_wealth"] <= cap_y2)]

m2, b2, r2, p2, _ = stats.linregress(renters_capped["financial_assets"], renters_capped["total_wealth"])
r2_full = renters["financial_assets"].corr(renters["total_wealth"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(renters_capped["financial_assets"], renters_capped["total_wealth"],
           alpha=0.35, color=RED, edgecolors="none", s=28, zorder=2)

ax.plot([0, cap_x2], [b2, m2 * cap_x2 + b2], color="darkred",
        linewidth=1.8, label=f"OLS fit  (r = {r2:.2f})", zorder=3)

ax.set_title("Financial Assets vs. Total Wealth - Renters")
ax.set_xlabel("Financial Assets (USD)")
ax.set_ylabel("Total Wealth (USD)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.set_xlim(0, cap_x2)
ax.set_ylim(0, cap_y2)
ax.legend(loc="upper left")
ax.annotate("Capped at 99th percentile", xy=(0.99, 0.03),
            xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7.5, color=GRAY, style="italic")
fig.text(0.5, -0.02,
         f"Figure 5: Scatter plot of financial assets vs. total wealth for renters (n = {len(renters_capped):,}, capped). "
         f"Pearson r = {r2:.2f}.",
         ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
plt.savefig("fig5_finassets_vs_wealth.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Pearson r on capped data (renters): {r2:.4f}")
print(f"Pearson r on full data   (renters): {r2_full:.4f}")

# %% [markdown]
# ## 7 . Median Wealth by Income Quintile and Tenure

# %%
df["income_quintile"] = pd.qcut(
    df["total_income"].clip(lower=1), q=5,
    labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])

quintile_wealth = (df.groupby(["income_quintile", "tenure"], observed=True)["total_wealth"]
                   .median().reset_index())

q_labels = ["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"]
x     = np.arange(len(q_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
for i, (tenure, color, label) in enumerate(
        [("Homeowner", BLUE, "Homeowner"), ("Renter", RED, "Renter")]):
    vals = [
        quintile_wealth.loc[
            (quintile_wealth["tenure"] == tenure) &
            (quintile_wealth["income_quintile"] == q), "total_wealth"
        ].values[0] if len(quintile_wealth.loc[
            (quintile_wealth["tenure"] == tenure) &
            (quintile_wealth["income_quintile"] == q)]) > 0 else 0
        for q in q_labels
    ]
    bars = ax.bar(x + i * width - width / 2, vals, width, label=label, color=color, zorder=3)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8000,
                    f"${val:,.0f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=color, rotation=45)

ax.set_title("Median Total Wealth by Income Quintile and Housing Tenure")
ax.set_xlabel("Income Quintile")
ax.set_ylabel("Median Total Wealth (USD)")
ax.set_xticks(x)
ax.set_xticklabels(q_labels)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.tick_params(axis="x", length=0)
ax.legend()
ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
fig.text(0.5, -0.02,
         "Figure 6: Median total wealth by income quintile and tenure. "
         "Homeowners maintain higher wealth than renters across all income levels.",
         ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
plt.savefig("fig6_wealth_by_quintile.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8 . Correlation Heatmap (Homeowners)

# %%
corr_cols = {
    "home_equity":         "Home Equity",
    "financial_assets":    "Financial Assets",
    "total_income":        "Total Income",
    "total_wealth":        "Total Wealth",
    "mortgage_balance":    "Mortgage Balance",
    "vehicle_value_total": "Vehicle Value",
    "business_value":      "Business Value",
}
corr_data   = owners[list(corr_cols.keys())].rename(columns=corr_cols)
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.4, linecolor="#dddddd",
            annot_kws={"size": 8.5},
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
            ax=ax)
ax.set_title("Correlation Matrix - Homeowners", pad=12)
ax.tick_params(axis="x", rotation=30, labelsize=8.5)
ax.tick_params(axis="y", rotation=0,  labelsize=8.5)
fig.text(0.5, -0.02,
         "Figure 7: Pearson correlation coefficients among key wealth variables for homeowners.",
         ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
plt.savefig("fig7_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9 . Share of Total Wealth Explained by Asset Type

# %%
owners["equity_share"] = (owners["home_equity"] /
                           owners["total_wealth"].replace(0, np.nan)).clip(0, 1)
owners["fin_share"]    = (owners["financial_assets"] /
                           owners["total_wealth"].replace(0, np.nan)).clip(0, 1)
renters["fin_share"]   = (renters["financial_assets"] /
                           renters["total_wealth"].replace(0, np.nan)).clip(0, 1)

owner_eq_med   = owners["equity_share"].median()
owner_fin_med  = owners["fin_share"].median()
renter_fin_med = renters["fin_share"].median()

print(f"Median equity share of wealth  (homeowners):   {owner_eq_med:.1%}")
print(f"Median fin-asset share of wealth (homeowners): {owner_fin_med:.1%}")
print(f"Median fin-asset share of wealth (renters):    {renter_fin_med:.1%}")

categories = ["Home Equity\n(Homeowners)", "Financial Assets\n(Homeowners)",
              "Financial Assets\n(Renters)"]
values     = [owner_eq_med, owner_fin_med, renter_fin_med]
bar_colors = [BLUE, LBLUE, RED]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(categories, values, color=bar_colors, width=0.45, zorder=3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{val:.1%}", ha="center", va="bottom",
            fontweight="bold", fontsize=10, color=bar.get_facecolor())

ax.set_title("Median Share of Total Wealth by Asset Type and Tenure")
ax.set_ylabel("Median Share of Total Wealth")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1.2)
ax.tick_params(axis="x", length=0)
fig.text(0.5, -0.02,
         "Figure 8: Median share of total wealth attributable to each asset type by tenure. "
         "Home equity accounts for the vast majority of homeowner wealth.",
         ha="center", fontsize=8, color=GRAY, style="italic")
plt.tight_layout()
plt.savefig("fig8_wealth_share.png", dpi=150, bbox_inches="tight")
plt.show()
