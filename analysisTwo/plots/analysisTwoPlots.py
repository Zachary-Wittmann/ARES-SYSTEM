import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

battle = pd.read_csv("data/battles.csv")
terrain = pd.read_csv("data/terrain.csv")
weather = pd.read_csv("data/weather.csv")

df = pd.merge(battle, terrain, on="isqno")
df = pd.merge(df, weather, on="isqno")
df.set_index("isqno", inplace=True)

# Fill NaN values in wina column with -1 based on research
df["wina"] = df["wina"].fillna(-1)
df = df[
    [
        "surpa",
        "post1",
        "post2",
        "wx1",
        "wx2",
        "wx3",
        "wx4",
        "wx5",
        "terra1",
        "terra2",
        "terra3",
        "aeroa",
        "wina",
    ]
]

surprise = df[["surpa", "wina"]]
terrain = df[["terra1", "terra2", "terra3", "wina"]]
weather = df[["wx1", "wx2", "wx3", "wx4", "wx5", "wina"]]
fortification = df[["post1", "post2", "wina"]]
aerial = df[["aeroa", "wina"]]

# # Plotting Element of Surprise Information
# plt.figure(figsize=(12, 12))
# sns.set_style("whitegrid")
# surprisePlot = sns.countplot(x="wina", hue="surpa", data=surprise, palette="Blues")
# surprisePlot.set_title("Element of Surprise Advantage")
# surprisePlot.set(
#     xlabel="Outcome",
#     ylabel="Count",
#     xticklabels=["Attacker Lost", "Draw", "Atacker Won"],
# )
# surprisePlot.legend(
#     title="Surprise Level",
#     labels=[
#         "Neither Side/No Affect",
#         "Minor by Attacker",
#         "Substantial by Attacker",
#         "Complete by Attacker",
#     ],
# )
# # plt.savefig("elementOfSurprise-countPlot.png")


# # Plotting Terrain1 Information
# plt.figure(figsize=(12, 12))
# sns.set_style("whitegrid")
# terrainPlot1 = sns.countplot(x="wina", hue="terra1", data=terrain, palette="Blues")
# terrainPlot1.set_title("Terrain 1")
# terrainPlot1.set(
#     xlabel="Outcome",
#     ylabel="Count",
#     xticklabels=["Attacker Lost", "Draw", "Atacker Won"],
# )
# terrainPlot1.legend(
#     title="Terrain",
#     labels=[
#         "Rolling",
#         "Rugged",
#         "Flat",
#     ],
# )
# # plt.savefig("terrain1-countPlot.png")

# # Plotting Terrain2 Information
# plt.figure(figsize=(12, 12))
# sns.set_style("whitegrid")
# terrainPlot2 = sns.countplot(x="wina", hue="terra2", data=terrain, palette="Blues")
# terrainPlot2.set_title("Terrain 2")
# terrainPlot2.set(
#     xlabel="Outcome",
#     ylabel="Count",
#     xticklabels=["Attacker Lost", "Draw", "Atacker Won"],
# )
# terrainPlot2.legend(
#     title="Terrain",
#     labels=[
#         "Bare",
#         "Mixed",
#         "Desert",
#         "Heavily Wooded",
#     ],
# )
# # plt.savefig("terrain2-countPlot.png")

# # Plotting Terrain3 Information
# plt.figure(figsize=(12, 12))
# sns.set_style("whitegrid")
# terrainPlot3 = sns.countplot(x="wina", hue="terra3", data=terrain, palette="Blues")
# terrainPlot3.set_title("Terrain 3")
# terrainPlot3.set(
#     xlabel="Outcome",
#     ylabel="Count",
#     xticklabels=["Attacker Lost", "Draw", "Atacker Won"],
# )
# terrainPlot3.legend(
#     title="Terrain",
#     labels=[
#         "Dunes",
#         "Urban",
#         "Marsh/Swamp",
#     ],
# )
# # plt.savefig("terrain3-countPlot.png")

# # Plotting Weather Information
# weatherPlotInfo = {
#     1: ["Conditions", "Dry", "Wet"],
#     2: [
#         "Precipitation Level",
#         "Sunny (No Precipitation)",
#         "Light Precipitation",
#         "Heavy Precipitation",
#         "Overcast (No Precipitation)",
#     ],
#     3: ["Temperature", "Temperate", "Hot", "Cold"],
#     4: ["Season", "Summer", "Winter", "Spring", "Autumn"],
#     5: ["Climate", "Temperate", "Tropical (Equatorial)", "Desert"],
# }

# for j in range(1, 6):
#     plt.figure(figsize=(12, 12))
#     sns.set_style("whitegrid")
#     weatherPlot = sns.countplot(
#         x="wina", hue="wx{}".format(j), data=weather, palette="Blues"
#     )
#     weatherPlot.set_title(f"Weather {j}")
#     weatherPlot.set(
#         xlabel="Outcome",
#         ylabel="Count",
#         xticklabels=["Attacker Lost", "Draw", "Atacker Won"],
#     )
#     weatherPlot.legend(title=weatherPlotInfo[j][0], labels=weatherPlotInfo[j][1:])
#     # plt.savefig(f"weather{j}-countPlot.png")

# # Plotting Defense Post1 Information
# plt.figure(figsize=(12, 12))
# sns.set_style("whitegrid")
# defensePost1Plot = sns.countplot(
#     x="wina", hue="post1", data=fortification, palette="Blues"
# )
# defensePost1Plot.set_title("Defense 1")
# defensePost1Plot.set(
#     xlabel="Outcome",
#     ylabel="Count",
#     xticklabels=["Attacker Lost", "Draw", "Atacker Won"],
# )
# defensePost1Plot.legend(
#     title="Type of Defense",
#     labels=[
#         "Hasty Defense",
#         "Prepared Defense",
#         "Fortified Defense",
#         "Delaying Action Adopted",
#         "Withdrawral Adopted",
#     ],
# )
# # plt.savefig("defense1-countPlot.png")

# # Plotting Defense Post2 Information
# plt.figure(figsize=(12, 12))
# sns.set_style("whitegrid")
# defensePost2Plot = sns.countplot(
#     x="wina", hue="post2", data=fortification, palette="Blues"
# )
# defensePost2Plot.set_title("Defense 2")
# defensePost2Plot.set(
#     xlabel="Outcome",
#     ylabel="Count",
#     xticklabels=["Attacker Lost", "Draw", "Atacker Won"],
# )
# defensePost2Plot.legend(
#     title="Type of Defense",
#     labels=[
#         "Prepared Defense",
#         "Fortified Defense",
#         "Withdrawral Adopted",
#         "Delaying Action Adopted",
#     ],
# )
# # plt.savefig("defense2-countPlot.png")

# # Plotting Aerial Superiority Information
# plt.figure(figsize=(12, 12))
# sns.set_style("whitegrid")
# aerialPlot = sns.countplot(x="wina", hue="aeroa", data=aerial, palette="Blues")

# aerialPlot.set(
#     xlabel="Outcome",
#     ylabel="Count",
#     xticklabels=["Attacker Lost", "Draw", "Atacker Won"],
# )
# aerialPlot.set_title("Aerial Superiority")
# aerialPlot.legend(
#     title="Aerial Superiority",
#     labels=[
#         "Neither Side",
#         "Attacker",
#     ],
# )
# # plt.savefig("areialSuperiority-countPlot.png")

# plt.show()
