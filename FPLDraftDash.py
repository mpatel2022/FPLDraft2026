import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import datetime
import requests
import plotly.express as px
import pickle
import os

# league id found by going to the end point: https://draft.premierleague.com/api/bootstrap-dynamic
league_id = 8918
url_all = 'https://draft.premierleague.com/api/bootstrap-static'
LOCAL_DIR = "/home/mpatel99/FPLDraft2026"
# LOCAL_DIR = "."

refresh_data = False

IMAGES_LOCATION = 'assets\\'

colour_grey_black = "#363434"
colour_white = "white"
colour_black = "#121212"
transparent = "rgba(0, 0, 0, 0)"

USER_MAP = {
    77347:'Salmon',
    71339:'Mitesh',
    36507:'Phil',
    102583:'Marcus',
    136961:'TomH',
    37121:'Kieran',
    109666:'Dan',
    108014:'Bipin',
    137445:'Rich',
}

user_ids = list(USER_MAP.keys())
user_names = [USER_MAP[user_id] for user_id in user_ids]


def discrete_background_color_bins(df, n_bins=5, columns='all', scale='Blues'):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq'][scale][i - 1]
        color = colour_white if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))


def get_shirt_logo_name(team_code):
    return f'shirt_{team_code}-66.webp'


def get_shirt_image(team_code):
    return f"/assets/{get_shirt_logo_name(team_code)}"


def save_image_data(image_url, image_name, folder):
    img_data = requests.get(f'{image_url}{image_name}').content
    with open(f'{folder}{image_name}', 'wb') as handler:
        handler.write(img_data)    


def get_all_gameweek_summary(df):
    # Total score per gameweek
    gw_scores = df.groupby("Gameweek")["Score"].sum()

    best_gw = gw_scores.idxmax()
    best_gw_score = gw_scores.max()

    worst_gw = gw_scores.idxmin()
    worst_gw_score = gw_scores.min()

    # Total score per player
    player_scores = df.groupby("Player")["Score"].sum()
    best_player = player_scores.idxmax()
    best_player_total = player_scores.max()

    # Best single gameweek score by any player
    player_gw_scores = df.groupby(["Player", "Gameweek"])["Score"].sum()
    best_player_gw = player_gw_scores.idxmax()
    best_player_gw_score = player_gw_scores.max()

    return {
        "Best Gameweek": best_gw,
        "Best GW Score": best_gw_score,
        "Worst Gameweek": worst_gw,
        "Worst GW Score": worst_gw_score,
        "Best Player": best_player,
        "Best Player Total": best_player_total,
        "Best Player GW": best_player_gw[1],
        "Best Player GW Score": best_player_gw_score
    }


def top_10_player_pie(df):
    top_players = df.groupby("player_name")["total_points"].sum().nlargest(10)
    fig = px.pie(
        names=top_players.index,
        values=top_players.values,
        title="Top 10 Player Contributions",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), 
        showlegend=False, 
        plot_bgcolor=transparent, 
        paper_bgcolor=colour_grey_black,
        font=dict(color=colour_white))
    return fig


def get_ranked_table_with_colours(standings_df):
    import colorlover as cl
    colors = cl.scales[f'{len(standings_df.columns)}']['div']['RdBu']

    style_data_conditional = []
    for col in standings_df.columns:  # Each gameweek column
        for n, color in enumerate(colors):
            style_data_conditional.append({
                "if": {
                    "column_id": col,
                    "filter_query": f"{{{col}}} = {n+1}"
                },
                "backgroundColor": color,
                "color": colour_black,
                "fontWeight": "bold",
            })

    style_data_conditional.append({
        "if": {
            "column_id": standings_df.reset_index().columns[0],
            "filter_query": "1 != 0"
        },
        "backgroundColor": colour_grey_black,
        "color": colour_white,
        "fontWeight": "bold",
    })

    table = dash_table.DataTable(
        data=standings_df.sort_index(ascending=False).reset_index().to_dict("records"),
        columns=[{"name": i, "id": i} for i in standings_df.reset_index().columns],
        style_data_conditional=style_data_conditional,
        style_cell={"textAlign": "center"},
        style_header={"backgroundColor": "#363434", "fontWeight": "bold"},
    )
    return table 



def get_aggregate_data_based_on_filter(merged_df, filter_type):
    starters_selection = merged_df["Position"] <= 11
    starters = merged_df[starters_selection]   
    if filter_type == 'all':
        agg = starters.groupby(['User', 'Gameweek'])['total_points'].sum().reset_index()
        title = "Total Points"
    elif filter_type == 'bonus':
        agg = starters.groupby(['User', 'Gameweek'])['bonus'].sum().reset_index()
        title = "Bonus Points Only"
        agg.columns = ['total_points']
    elif filter_type == 'subs':
        agg = merged_df[~starters_selection].groupby(['User', 'Gameweek'])['total_points'].sum().reset_index()
        title = "Subs Only"        
    elif filter_type in ['GKP', 'DEF', 'MID', 'FWD']:
        filtered = starters[starters['position'] == filter_type]
        title = f"Total Points - {filter_type}"
        agg = filtered.groupby(['User', 'Gameweek'])['total_points'].sum().reset_index()
    agg = agg.pivot(index='Gameweek', columns='User', values='total_points')
    return agg, title    


if refresh_data:
    r = requests.get(url_all)
    all_data = r.json()

    with open(f'{LOCAL_DIR}/metadata.pickle', 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Get the team logos and store in resources
    shirt_logos_url = "https://fantasy.premierleague.com/dist/img/shirts/standard/"
    team_badges_url = "https://resources.premierleague.com/premierleague25/badges/"
    for i in all_data['teams']:
        team_code = i['code']

        shirt_logo_name = get_shirt_logo_name(team_code)
        save_image_data(shirt_logos_url, shirt_logo_name, IMAGES_LOCATION)
        save_image_data(team_badges_url, f'{team_code}.webp', IMAGES_LOCATION)        

with open(f'{LOCAL_DIR}/metadata.pickle', 'rb') as handle:
    all_data = pickle.load(handle)    

next_gameweek = all_data['events']['next']
gameweeks = np.arange(next_gameweek)[1:]

# Mapping
# Teams
team_map = pd.DataFrame({i['id']: [i['short_name'], 
                                   i['name'],
                                   i['code']]  
                        for i in all_data['teams']}).T
team_map.columns = ['team_short_name', 'team_name', 'team_code']

player_map = pd.DataFrame({i['id']: ['{} {}'.format(i['first_name'],i['second_name']), 
                                     i['web_name'],
                                     i['code'],
                                     i['team'],
                                     i['element_type']]  
                          for i in all_data['elements']}).T
player_map.columns = ['full_name', 'web_name', 'player_code', 'team_id', 'position_id']
all_players = player_map.index.tolist()


positions_map = pd.Series({i['id']:i['plural_name_short'] 
                          for i in all_data['element_types']},
                          name='position')

player_map = pd.merge(player_map, 
                      positions_map,
                      left_on=['position_id'],
                      right_index=True).rename(columns={'position_y': 'position'})

player_map = pd.merge(player_map, 
                      team_map,
                      left_on=['team_id'],
                      right_index=True)

player_map['player_name'] = [f'{x} ({y})' for x, y in zip(player_map['web_name'], player_map['team_short_name'])]

if refresh_data:
    player_history = []
    for player in all_players:
        url_player = 'https://draft.premierleague.com/api/element-summary/{}'.format(player)
        r = requests.get(url_player)
        
        if r.status_code == 502:
            continue
        player_data_json = r.json()['history']   
        data_length = len(player_data_json)
        for i in np.arange(data_length):
            player_history = player_history + [pd.Series(player_data_json[i])]
    player_history = pd.concat(player_history,axis=1).T.rename(columns={'event':'gameweek'})        
    player_history.to_pickle('player_history.pickle')    

    url_league = 'https://draft.premierleague.com/api/league/{}/details'.format(league_id)
    r = requests.get(url_league)
    league_data = r.json()
    with open(f'{LOCAL_DIR}/league_data.pickle', 'wb') as handle:
        pickle.dump(league_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for gameweek in gameweeks:
        filename = f'team_positions_history_gw{gameweek}.pickle'
        if not os.path.exists(filename):
            team_positions_array = []
            for user_id in user_ids:
                url_team = 'https://draft.premierleague.com/api/entry/{}/event/{}'.format(user_id, gameweek)
                r = requests.get(url_team)
                
                team_data_json = r.json()
                team_positions = pd.Series({team_data_json['picks'][i]['position']:team_data_json['picks'][i]['element'] for i in np.arange(len(team_data_json['picks']))})
                team_positions['user_id'] = user_id
                team_positions['gameweek'] = gameweek
                                    
                team_positions_array.append(team_positions)
            team_positions_history = pd.concat(team_positions_array,axis=1).T
            team_positions_history = team_positions_history.set_index(['user_id','gameweek']).stack().reset_index().rename(columns={'level_2':'position',0:'element'})
            team_positions_history.to_pickle(f'team_positions_history_gw{gameweek}.pickle')


with open(f'{LOCAL_DIR}/league_data.pickle', 'rb') as handle:
    league_data = pickle.load(handle)    

with open(f'{LOCAL_DIR}/player_history.pickle', 'rb') as handle:
    player_df = pickle.load(handle)    

team_array = []
for gameweek in gameweeks:
    with open(f'{LOCAL_DIR}/team_positions_history_gw{gameweek}.pickle', 'rb') as handle:
        team_array.append(pickle.load(handle))    

team_df = pd.concat(team_array)


# -------------------------------
# 📦 Load and Prepare the Data
# -------------------------------
team_df['user_name'] = team_df['user_id'].map(USER_MAP)
team_df.rename(columns={"user_name": "User", 
                        "gameweek": "Gameweek", 
                        "position": "Position", 
                        "element": "PlayerID"}, inplace=True)

player_df.rename(columns={"element": "PlayerID", 
                          "gameweek": "Gameweek", 
                          "web_name": "name"}, inplace=True)

player_map.rename(columns={"web_name": "name"}, inplace=True)


# Merge datasets
merged_df = pd.merge(team_df, player_df, on=["PlayerID", "Gameweek"], how="left")
merged_df = pd.merge(merged_df, player_map, left_on=["PlayerID"], right_index=True, how="left")
merged_df.sort_values(by=["User", "Gameweek", "Position"], inplace=True)
merged_df = pd.DataFrame(merged_df.to_dict())

# -------------------------------
# 🚀 Initialize Dash App
# -------------------------------

# app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server  # for deployment

# Dropdown options
user_options = sorted(merged_df["User"].unique())
gameweek_options = sorted(merged_df["Gameweek"].unique(), reverse=True)

# -------------------------------
# 🎨 App Layout
# -------------------------------
app.layout = html.Div([
    html.H2("FPL Draft 2025/26", style={"textAlign": "center", "color": "white"}),

    dcc.Tabs(
        id="tabs", 
        value="all-users-view", 
        children=[
            dcc.Tab(label="All Users Stats", value="all-users-view"),
            dcc.Tab(label="User Stats", value="single-user-view")
        ],
    ),

    html.Div(id="tab-content", style={"padding": "20px"})
])

# -------------------------------
# 📊 Callback for Pitch + Summary
# -------------------------------
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab(tab):
    if tab == "all-users-view":
        tab_data = html.Div([
            html.H4("Performance Stats (All Gameweeks)"),
            dcc.Tabs(
                id="stats-tabs", 
                value="standings-tab", 
                children=[
                    dcc.Tab(label="Standings", value="standings-tab"),
                    dcc.Tab(label="Performance", value="performance-tab"),
                    dcc.Tab(label="Squad", value="squad-tab")
                ]
            ),
            html.Div(id="stats-tab-content", style={"marginTop": "20px"})
        ])
    elif tab == "single-user-view":
        tab_data = html.Div([
            html.Div([
                html.Label("Select User:", style={"color": "white"}),
                dcc.Dropdown(
                    id="user-dropdown",
                    options=[{"label": str(u), "value": u} for u in user_options],
                    value=user_options[0],
                    clearable=False
                ),
            ], style={"width": "48%", "display": "inline-block"}),

            html.Div([
                html.Label("Select Gameweek:", style={"color": "white"}),
                dcc.Dropdown(
                    id="gameweek-dropdown",
                    options=[{"label": f"GW {gw}", "value": gw} for gw in gameweek_options],
                    value=gameweek_options[0],
                    clearable=False
                )], style={"width": "48%", "display": "inline-block", "float": "right"}),
            html.Br(),
            html.Br(),
            html.Div([
                html.Div([
                    html.H4("Team"),
                    dcc.Graph(id="pitch-graph")], className="graph-container",
                    style={"width": "50%", "paddingLeft": "20px", "paddingRight": "20px", "display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    html.H4("Gameweek Summary"),
                    html.Div(className="summary-panel", id="summary-panel"),
                    html.Br(),
                    html.Div([dcc.Graph(id="pie-chart")], className="graph-container")
                ], style={"width": "48%", "display": "inline-block", "paddingLeft": "20px"})
            ])
        ])
    return tab_data


@app.callback(
    Output("stats-tab-content", "children"),
    Input("stats-tabs", "value")
)
def render_stats_subtab(active_tab):
    if active_tab == "standings-tab":
        return html.Div([
            html.H5("Standings Overview", style={"color": "white"}),
            dcc.RadioItems(
                className="dash-radio-items",
                id="standings-toggle",
                options=[
                    {"label": "Chart", "value": "chart"},
                    {"label": "Table", "value": "table"}
                ],
                value="table",
                labelStyle={"display": "inline-block", "marginRight": "10px"},
                style={"marginTop": "5px"},
            ),
            html.Br(),
            html.Div(id="standings-content")
        ])
    elif active_tab == "performance-tab":
        return html.Div([
                html.Div([
                    html.H5("Points per gameweek", style={"marginTop": "20px"}),
                    dcc.Dropdown(
                        id='performance-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'GKP', 'value': 'GKP'},
                            {'label': 'DEF', 'value': 'DEF'},                        
                            {'label': 'MID', 'value': 'MID'},
                            {'label': 'FWD', 'value': 'FWD'},                                                
                            {'label': 'Bonus Points Only', 'value': 'bonus'}
                        ],
                        value='all',
                        clearable=False,
                        style={"width": "300px", "marginTop": "5px"},
                    ),
                    dcc.RadioItems(
                        className="dash-radio-items",
                        id="performance-toggle",
                        options=[
                            {"label": "Chart", "value": "chart"},
                            {"label": "Table", "value": "table"}
                        ],
                        value="table",
                        labelStyle={"display": "inline-block", "marginRight": "10px"},
                        style={"marginTop": "5px"},
                    ),
                    html.Br(),
                    html.Div(id="performance-content")
                ]),
                html.Br(),
                html.Div([
                    html.H5("Cumulative points", style={"marginTop": "20px"}),
                    dcc.Dropdown(
                        id='cumulative-performance-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'GKP', 'value': 'GKP'},
                            {'label': 'DEF', 'value': 'DEF'},                        
                            {'label': 'MID', 'value': 'MID'},
                            {'label': 'FWD', 'value': 'FWD'},                                                
                            {'label': 'Bonus Points Only', 'value': 'bonus'}
                        ],
                        value='all',
                        clearable=False,
                        style={"width": "300px", "marginTop": "5px"}
                    ),
                    dcc.RadioItems(
                        className="dash-radio-items",
                        id="cumulative-performance-toggle",
                        options=[
                            {"label": "Chart", "value": "chart"},
                            {"label": "Table", "value": "table"}
                        ],
                        value="chart",
                        labelStyle={"display": "inline-block", "marginRight": "10px", "marginTop": "5px"},
                        style={"marginTop": "5px"},
                    ),
                    html.Br(),
                    html.Div(id="cumulative-performance-content")
                ]),
                html.Br(),
                html.Div([
                    html.H5("Performance spread", style={"marginTop": "20px"}),
                    dcc.RadioItems(
                        className="dash-radio-items",
                        id="performance-spread-toggle",
                        options=[
                            {"label": "Mean", "value": "mean"},
                            {"label": "Spread", "value": "spread"}
                        ],
                        value="mean",
                        labelStyle={"display": "inline-block", "marginRight": "10px", "marginTop": "5px"},
                        style={"marginTop": "5px"},
                    ),
                    html.Div(id="performance-spread-content")
                ]),                
                html.Br(),
                html.Div([
                    html.H5("Rolling n-week average", style={"marginTop": "20px"}),
                    dcc.Dropdown(
                        id='rolling-performance-filter',
                        options=[
                            {'label': '2-week', 'value': 2},
                            {'label': '3-week', 'value': 3},
                            {'label': '4-week', 'value': 4},
                            {'label': '5-week', 'value': 5},
                            {'label': '6-week', 'value': 6},
                            {'label': '7-week', 'value': 7},
                            {'label': '8-week', 'value': 8}
                        ],
                        value=4,
                        clearable=False,
                        style={"width": "300px", "marginTop": "5px"}
                    ),
                    html.Div(id="rolling-performance-content")
                ]),
                html.Br(),                
            ])
    elif active_tab == "squad-tab":
        return html.Div([
                html.Div([
                    html.H5("Squad Stats"),
                    dcc.Dropdown(
                                id='n-players-used-filter',
                                options=[
                                    {'label': 'Starters', 'value': 'starters'},
                                    {'label': 'GKP', 'value': 'GKP'},
                                    {'label': 'DEF', 'value': 'DEF'},                        
                                    {'label': 'MID', 'value': 'MID'},
                                    {'label': 'FWD', 'value': 'FWD'},                                                
                                    {'label': 'All Players', 'value': 'all'}
                                ],
                                value='starters',
                                clearable=False,
                                style={"width": "300px", "marginTop": "5px"}
                            ),
                    html.Div(id="n-players-used-content"),
                    html.Br(),
            ])
        ])


@app.callback(
    Output("standings-content", "children"),
    Input("standings-toggle", "value")
)
def update_standings_view(view_mode):
    standings_df = merged_df[merged_df['Position'] <= 11].groupby(["User", "Gameweek"])["total_points"].sum().reset_index()
    standings_df = (
        standings_df.pivot(index="Gameweek", columns="User", values="total_points").fillna(0)
        .sort_index()
        .cumsum()
        .sort_index(ascending=False)
        .rank(1, ascending=False, method='min')
        .astype(int)
    )

    if view_mode == "chart":
        fig = px.line(standings_df, title='Position', labels={'User':''})
        fig.update_layout(
            plot_bgcolor=transparent, 
            paper_bgcolor=transparent,
            font=dict(color=colour_white),
            margin=dict(t=40, b=20, l=20, r=20)
        )
        return html.Div([dcc.Graph(figure=fig)], className="graph-container")

    else:  # table view
        table = get_ranked_table_with_colours(standings_df)
        return table


@app.callback(
    [Output("pitch-graph", "figure"),
     Output("pie-chart", "figure"),
     Output("summary-panel", "children")],
    [Input("user-dropdown", "value"),
     Input("gameweek-dropdown", "value")]
)
def update_pitch(user, gw):
    gw = gameweek_options[-2] if gw == 'All' else gw
    df = merged_df[(merged_df["User"] == user) & (merged_df["Gameweek"] == gw)]

    starters = df[df["Position"] <= 11].copy()
    subs = df[df["Position"] > 11].copy()

    pitch_width = 100
    pitch_length = 100
    pitch_sub_length = -20

    def assign_coordinates(df):
        layout = {}
        lines = {
            "GKP": (pitch_length / 10, 1),
            "DEF": (3.5 * pitch_length / 10, 5),
            "MID": (6.2 * pitch_length / 10, 5),
            "FWD": (8.7 * pitch_length / 10, 3)
        }

        for role, (y, default_count) in lines.items():
            players = df[df["position"] == role]
            count = len(players)
            if count == 0:
                []
            elif count == 1:
                xs = [pitch_width/2]
            else:
                margin = pitch_width / default_count 
                xs = list(np.linspace(margin, pitch_width - margin, count))            

            for i, (idx, row) in enumerate(players.iterrows()):
                layout[idx] = (xs[i], y)
        return layout

    starter_coords = assign_coordinates(starters)
    starters["x"] = starters.index.map(lambda i: starter_coords[i][0])
    starters["y"] = starters.index.map(lambda i: starter_coords[i][1])

    sub_margin = pitch_width / 5
    subs["x"] = [sub_margin, 2 * sub_margin, 3 * sub_margin, 4 * sub_margin]
    subs["y"] = [pitch_sub_length / 2] * len(subs)

    # Pitch background stripes
    pitch_shapes = []
    for i in range(0, pitch_length, 10):
        color = "#228B22" if (i // 10) % 2 == 0 else "#32CD32"
        pitch_shapes.append(dict(type="rect", x0=0, x1=pitch_width, y0=i, y1=i+10, fillcolor=color, line=dict(width=0), layer="below"))

    # Pitch markings
    pitch_shapes += [
        dict(type="line", x0=0, x1=pitch_width, y0=(95*pitch_length/100), y1=(95*pitch_length/100), line=dict(color=colour_white, width=2)),  # halfway
        dict(type="circle", x0=4*pitch_width/10, x1=6*pitch_width/10, y0=(85*pitch_length/100), y1=(110*pitch_length/100), line=dict(color=colour_white, width=2)),  # center circle
        dict(type="rect", x0=3*pitch_width/10, x1=7*pitch_width/10, y0=0, y1=20*pitch_length/100, line=dict(color=colour_white, width=2)),  # penalty box
        dict(type="rect", x0=4*pitch_width/10, x1=6*pitch_width/10, y0=0, y1=10*pitch_length/100, line=dict(color=colour_white, width=2)),  # goal area
    ]

    # Image annotations
    images = []
    for _, row in pd.concat([starters, subs]).iterrows():
        images.append(dict(
            source=get_shirt_image(row["team_code"]),
            x=row["x"]-3, y=row["y"] - 2,
            xref="x", yref="y",
            sizex=10, sizey=10,
            xanchor="left", yanchor="bottom",
            layer="above"
        ))

    # Text annotations
    texts = []
    for _, row in pd.concat([starters, subs]).iterrows():
        label = f"<b>{row['name']}</b><br><b>{row['total_points']}</b>"
        texts.append(dict(x=row["x"], y=row["y"]-6.0, text=label, showarrow=False,
                          font=dict(size=10, color=colour_white), bgcolor='#175717', align="center"))

    # Build figure
    fig = go.Figure()
    fig.update_layout(
        shapes=pitch_shapes,
        images=images,
        annotations=texts,
        xaxis=dict(range=[0, pitch_width], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[pitch_sub_length, pitch_length], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor=colour_grey_black, 
        paper_bgcolor=transparent,    
        font=dict(color=colour_white),
        margin=dict(t=pitch_length/10, b=pitch_length/10, l=pitch_width/10, r=pitch_width/10),
        height=550
    )

    # Summary stats
    summary = html.Ul([
        html.Li(f"Total Starter Points: {starters['total_points'].sum()}"),
        html.Li(f"Total Sub Points: {subs['total_points'].sum()}"),
        html.Li(f"Goals Scored: {df['goals_scored'].sum()}"),
        html.Li(f"Assists: {df['assists'].sum()}"),
        html.Li(f"Clean Sheets: {df['clean_sheets'].sum()}")
    ])

    pie_fig = top_10_player_pie(starters)

    return fig, pie_fig, summary


def get_player_usage_data(df, filter_type):
    if filter_type == "all":
        agg = df.groupby("User")["PlayerID"].nunique()
        title = "Total Unique Players Used"
    elif filter_type in ("GKP", "DEF", "MID", "FWD"):
        agg = df[(df.position == filter_type) & (df.Position <= 11)].groupby("User")["PlayerID"].nunique()
        title = f"Total Unique {filter_type}s Used"
    else:
        agg = df[df.Position <= 11].groupby("User")["PlayerID"].nunique()
        title = "Total Unique Starters Used"
    return agg, title


@app.callback(
    Output('n-players-used-content', 'children'),
    Input('n-players-used-filter', 'value')
)
def update_n_players_used_graph(filter_type):
    agg, title = get_player_usage_data(merged_df, filter_type)
    agg = agg.sort_values(ascending=False)

    fig = px.bar(agg, labels={'value':'Number of unique players','user_name':''})

    fig.update_layout(
        title=title,
        yaxis_title="Players Used",
        plot_bgcolor=transparent,
        paper_bgcolor=transparent,
        showlegend=False,
        font=dict(color="#e0e0e0")
    )

    return html.Div([dcc.Graph(figure=fig)], className="graph-container")


@app.callback(
    Output('performance-content', 'children'),
    [Input('performance-filter', 'value'),
     Input("performance-toggle", "value")]
)
def update_performance_graph(filter_type, view_mode):
    agg, title = get_aggregate_data_based_on_filter(merged_df, filter_type)
    if view_mode == "chart":
        fig = px.line(agg, markers=True)

        fig.update_layout(
            title=title,
            xaxis_title="Gameweek",
            yaxis_title="Points",
            plot_bgcolor=transparent, 
            paper_bgcolor=transparent,
            font=dict(color=colour_white),
        )
        return html.Div([dcc.Graph(figure=fig)], className="graph-container")
    else:  # table view
        table = get_ranked_table_with_colours(agg.sort_index(ascending=False).rank(1, ascending=False, method='min'))
        return table


@app.callback(
    Output('rolling-performance-content', 'children'),
    Input('rolling-performance-filter', 'value')
)
def update_rolling_performance_graph(n_weeks):
    agg, title = get_aggregate_data_based_on_filter(merged_df, 'all')
    agg = agg.rolling(window=n_weeks, min_periods=1).sum()
    fig = px.line(agg, markers=True)
    fig.update_layout(
        title=f'Rolling {n_weeks}-average',
        xaxis_title="Gameweek",
        yaxis_title="Points",
        plot_bgcolor=transparent, 
        paper_bgcolor=transparent,
        font=dict(color=colour_white),
    )
    return html.Div([dcc.Graph(figure=fig)], className="graph-container")


@app.callback(
    Output('performance-spread-content', 'children'),
    Input('performance-spread-toggle', 'value')
)
def update_performance_spread_graph(view_mode):
    agg, title = get_aggregate_data_based_on_filter(merged_df, 'all')
    
    if view_mode == 'mean':
        player_points = agg.mean(0).to_frame()
        player_points_std = agg.std(0).to_frame()
        player_points = pd.merge(player_points,
                                 player_points_std,
                                 right_index=True,
                                 left_index=True).reset_index()

        player_points.columns = ['User', 'Points', 'err']

        fig = px.scatter(player_points, 
                         x='User',
                         y='Points', 
                         error_y="err")
        fig.update_traces(marker_size=10)
        fig.update_xaxes(showgrid=False)
    else:
        fig = px.strip(agg)

    fig.update_layout(
        plot_bgcolor=transparent, 
        paper_bgcolor=transparent,
        yaxis_title="Points",
        font=dict(color=colour_white),
    )        
    return html.Div([dcc.Graph(figure=fig)], className="graph-container")    


@app.callback(
    Output('cumulative-performance-content', 'children'),
    [Input('cumulative-performance-filter', 'value'),
     Input("cumulative-performance-toggle", "value")]
)
def update_cumulative_performance_graph(filter_type, view_mode):
    agg, title = get_aggregate_data_based_on_filter(merged_df, filter_type)
    agg = agg.cumsum()
    if view_mode == "chart":
        fig = px.line(agg, markers=True)

        fig.update_layout(
            title=title,
            xaxis_title="Gameweek",
            yaxis_title="Cumulative Points",
            plot_bgcolor=transparent, 
            paper_bgcolor=transparent,
            font=dict(color=colour_white),
        )
        return html.Div([dcc.Graph(figure=fig)], className="graph-container")

    else:  # table view
        table = get_ranked_table_with_colours(agg.sort_index(ascending=False).rank(1, ascending=False, method='min'))
        return table


# -------------------------------
# 🏁 Run the App
# -------------------------------

if __name__ == "__main__":
    app.run()