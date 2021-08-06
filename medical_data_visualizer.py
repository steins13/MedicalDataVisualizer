import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")


# Add 'overweight' column
# bmi
bmi = df["weight"] / (df["height"] * (10 ** -2)) ** 2
df["overweight"] = bmi
# assign 1 and 0 / convert to integer
df.loc[df["overweight"] <= 25, "overweight"] = 0
df.loc[df["overweight"] > 25, "overweight"] = 1
df["overweight"] = df["overweight"].astype(int)


# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.loc[df["cholesterol"] == 1, "cholesterol"] = 0
df.loc[df["cholesterol"] > 1, "cholesterol"] = 1
df.loc[df["gluc"] == 1, "gluc"] = 0
df.loc[df["gluc"] > 1, "gluc"] = 1


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars="cardio", value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat["total"] = 1
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index=False).count()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x="variable", y="total", kind="bar", data=df_cat, col="cardio", hue="value").fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.drop(df[(df["ap_hi"] < df["ap_lo"]) | (df["height"] < df["height"].quantile(0.025)) | (df["height"] > df["height"].quantile(0.975)) | (df["weight"] < df["weight"].quantile(0.025)) | (df["weight"] > df["weight"].quantile(0.975))].index)

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the heatmap with 'sns.heatmap()'
        ax = sns.heatmap(corr, mask=mask, center=0, vmin=-0.16, vmax=0.32, square=True, annot=True, fmt=".1f", cmap="icefire")

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
