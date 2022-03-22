import pandas as pd
from pulp import LpVariable, LpProblem, LpMaximize, LpStatus, lpSum

cricket_df = pd.read_csv("rcb2022.csv")


def clean_data(df_features):
    df = df_features.copy()

    df['quantity'] = 1

    df['selection_popularity'] = df['SelectionPercent'].apply(
        lambda x: int(x[:-1:]))
    df['selection_popularity'] = (df['selection_popularity'] *
                                  df['Price(IndCrores)']) / 100

    df = pd.get_dummies(df, columns=['Leader', 'PlayerRole', 'Nationality'])

    PlayerName = list(df['PlayerName'])
    player_dict = {}
    for col in df.columns:
        player_dict[col] = dict(zip(PlayerName, df[col].values))
    return PlayerName, player_dict


def team_optimization(player_name, features: dict):

    player_chosen = LpVariable.dict("Player Chosen",
                                    player_name,
                                    0,
                                    1,
                                    cat=int)

    # Title of LP
    prob = LpProblem("Optimize_Playing11", LpMaximize)

    # Objective Function
    prob += lpSum([
        features['selection_popularity'][i] * player_chosen[i]
        for i in player_name
    ]), "Maximize Points based on Selection Popularity"

    # Total Selection
    prob += lpSum([
        features['quantity'][f] * player_chosen[f] for f in player_name
    ]) == 11, "Playing 11"

    # Captain
    prob += lpSum([
        features['Leader_Captain'][f] * player_chosen[f] for f in player_name
    ]) == 1, "Captain"

    # Vice Captain
    prob += lpSum([
        features['Leader_ViceCaptain'][f] * player_chosen[f]
        for f in player_name
    ]) == 1, "Vice Captain"

    # Max and Min AllRounder
    prob += lpSum([
        features['PlayerRole_AllRounder'][f] * player_chosen[f]
        for f in player_name
    ]) <= 3, "Max AllRounder"
    prob += lpSum([
        features['PlayerRole_AllRounder'][f] * player_chosen[f]
        for f in player_name
    ]) >= 2, "Min AllRounder"

    # Max and Min Batsmen
    prob += lpSum([
        features['PlayerRole_Batsmen'][f] * player_chosen[f]
        for f in player_name
    ]) <= 5, "Max Batsmen"
    prob += lpSum([
        features['PlayerRole_Batsmen'][f] * player_chosen[f]
        for f in player_name
    ]) >= 3, "Min Batsmen"

    # Max and Min Bowler
    prob += lpSum([
        features['PlayerRole_Bowler'][f] * player_chosen[f]
        for f in player_name
    ]) <= 5, "Max Bowler"
    prob += lpSum([
        features['PlayerRole_Bowler'][f] * player_chosen[f]
        for f in player_name
    ]) >= 3, "Min Bowler"

    # Wicketkeeper
    prob += lpSum([
        features['PlayerRole_WicketKeeper'][f] * player_chosen[f]
        for f in player_name
    ]) >= 1, "Min WicketKeeper"

    # Indian Players
    prob += lpSum([
        features['Nationality_Indian'][f] * player_chosen[f]
        for f in player_name
    ]) <= 10, "Max Indian Players"
    prob += lpSum([
        features['Nationality_Indian'][f] * player_chosen[f]
        for f in player_name
    ]) >= 7, "Min Indian Players"

    # Foreign Players
    prob += lpSum([
        features['Nationality_Overseas'][f] * player_chosen[f]
        for f in player_name
    ]) <= 4, "Max Overseas Players"
    prob += lpSum([
        features['Nationality_Overseas'][f] * player_chosen[f]
        for f in player_name
    ]) >= 1, "Min Overseas Players"

    LpFile = prob.writeLP("Playing11_Cricket")

    prob.solve()

    return prob


player_names, feature = clean_data(cricket_df)
prob = team_optimization(player_names, feature)

player_roles = dict(zip(cricket_df['PlayerName'], cricket_df['PlayerRole']))

players_choosen = []
for player in prob.variables():
    if player.varValue > 0:
        act_name = " ".join(player.name.split('_')[-2:])
        players_choosen.append(act_name)

result_df = cricket_df[cricket_df['PlayerName'].isin(players_choosen)]
result_df.sort_values('Leader')
result_df.to_csv('RCBPlayingXI.csv', index=None)
