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

# For html output use:
# jupyter nbconvert --execute --no-input --no-prompt --to html FPLDraft.ipynb

# league id found by going to the end point: https://draft.premierleague.com/api/bootstrap-dynamic
league_id = 43259
url_all = 'https://draft.premierleague.com/api/bootstrap-static'

max_gameweek = 38

refresh_core_data = False

IMAGES_LOCATION = 'assets\\'

colour_grey_black = "#363434"
colour_white = "white"
colour_black = "#121212"
transparent = "rgba(0, 0, 0, 0)"


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


if refresh_core_data:
    r = requests.get(url_all)
    all_data = r.json()

    with open('metadata.pickle', 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Get the team logos and store in resources
    shirt_logos_url = "https://fantasy.premierleague.com/dist/img/shirts/standard/"
    team_badges_url = "https://resources.premierleague.com/premierleague25/badges/"
    for i in all_data['teams']:
        team_code = i['code']

        shirt_logo_name = get_shirt_logo_name(team_code)
        save_image_data(shirt_logos_url, shirt_logo_name, IMAGES_LOCATION)
        save_image_data(team_badges_url, f'{team_code}.webp', IMAGES_LOCATION)        

with open('metadata.pickle', 'rb') as handle:
    all_data = pickle.load(handle)    

# Mapping
# Teams
team_map = pd.DataFrame({i['id']: [i['short_name'], 
                                   i['name'],
                                   i['code']]  
                        for i in all_data['teams']}).T
team_map.columns = ['team_short_name', 'team_name', 'team_code']

# Players
player_map = pd.DataFrame({i['id']: ['{} {}'.format(i['first_name'],i['second_name']), 
                                     i['web_name'],
                                     i['code'],
                                     i['team'],
                                     i['element_type']]  
                          for i in all_data['elements']}).T
player_map.columns = ['full_name', 'web_name', 'player_code', 'team_id', 'position_id']


positions_map = pd.Series({i['id']:i['plural_name_short'] 
                          for i in all_data['element_types']},
                          name='position')

player_map = pd.merge(player_map, 
                      positions_map,
                      left_on=['position_id'],
                      right_index=True)

player_map = pd.merge(player_map, 
                      team_map,
                      left_on=['team_id'],
                      right_index=True)

player_map['player_name'] = [f'{x} ({y})' for x, y in zip(player_map['web_name'], player_map['team_short_name'])]

# -------------------------------
# 📦 Load and Prepare the Data
# -------------------------------

# Load team positions
team_df = pd.read_csv("team_positions_history.csv", index_col=0)

# Load player history
player_df = pd.read_csv("player_history.csv", index_col=0)
player_df = pd.merge(player_df, player_map, left_on=["element"], right_index=True, how="left")

user_map = {
    162095:'Salmon',
    266672:'Mitesh',
    162019:'Phil',
    206655:'Marcus',
    281930:'TomH',
    165434:'Kieran',
    277102:'Dan',
    278897:'Bipin',
}

user_ids = list(user_map.keys())
user_names = [user_map[user_id] for user_id in user_ids]

team_df['user_name'] = team_df['user_id'].map(user_map)
team_df.rename(columns={"user_name": "User", 
                        "gameweek": "Gameweek", 
                        "position": "Position", 
                        "element": "PlayerID"}, inplace=True)

player_df.rename(columns={"element": "PlayerID", 
                          "gameweek": "Gameweek", 
                          "web_name": "name"}, inplace=True)

# Merge datasets
merged_df = pd.merge(team_df, player_df, on=["PlayerID", "Gameweek"], how="left")
merged_df.sort_values(by=["User", "Gameweek", "Position"], inplace=True)

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
                                value='all',
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
    standings_df = merged_df.groupby(["User", "Gameweek"])["total_points"].sum().reset_index()
    standings_df["Cumulative Points"] = standings_df.groupby("User")["total_points"].cumsum()

    standings_df = (
        standings_df.pivot(index="Gameweek", columns="User", values="Cumulative Points").fillna(0)
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

    def assign_coordinates(df):
        layout = {}
        lines = {
            "GKP": (10, 1),
            "DEF": (27, 5),
            "MID": (45, 5),
            "FWD": (63, 3)
        }

        for role, (y, default_count) in lines.items():
            players = df[df["position"] == role]
            count = len(players)
            if count == 0:
                []
            elif count == 1:
                xs = [50]
            else: 
                xs = list(np.linspace(20, 80, count))            

            for i, (idx, row) in enumerate(players.iterrows()):
                layout[idx] = (xs[i], y)
        return layout

    starter_coords = assign_coordinates(starters)
    starters["x"] = starters.index.map(lambda i: starter_coords[i][0])
    starters["y"] = starters.index.map(lambda i: starter_coords[i][1])

    subs["x"] = [20, 40, 60, 80][:len(subs)]
    subs["y"] = [-8] * len(subs)

    # Pitch background stripes
    pitch_shapes = []
    for i in range(0, 70, 10):
        color = "#228B22" if (i // 10) % 2 == 0 else "#32CD32"
        pitch_shapes.append(dict(type="rect", x0=0, x1=100, y0=i, y1=i+10, fillcolor=color, line=dict(width=0), layer="below"))

    # Pitch markings
    pitch_shapes += [
        dict(type="line", x0=0, x1=100, y0=65, y1=65, line=dict(color=colour_white, width=2)),  # halfway
        dict(type="circle", x0=40, x1=60, y0=55, y1=80, line=dict(color=colour_white, width=2)),  # center circle
        dict(type="rect", x0=30, x1=70, y0=0, y1=15, line=dict(color=colour_white, width=2)),  # penalty box
        dict(type="rect", x0=40, x1=60, y0=0, y1=5, line=dict(color=colour_white, width=2)),  # goal area
    ]

    # Image annotations
    images = []
    for _, row in pd.concat([starters, subs]).iterrows():
        images.append(dict(
            source=get_shirt_image(row["team_code"]),
            x=row["x"]-2, y=row["y"] - 2,
            xref="x", yref="y",
            sizex=7, sizey=7,
            xanchor="left", yanchor="bottom",
            layer="above"
        ))

    # Text annotations
    texts = []
    for _, row in pd.concat([starters, subs]).iterrows():
        label = f"<b>{row['name']}</b><br><b>{row['total_points']}</b>"
        texts.append(dict(x=row["x"], y=row["y"]-5.5, text=label, showarrow=False,
                          font=dict(size=9, color=colour_white), bgcolor='#175717', align="center"))

    # Build figure
    fig = go.Figure()
    fig.update_layout(
        shapes=pitch_shapes,
        images=images,
        annotations=texts,
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-18, 70], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor=colour_grey_black, 
        paper_bgcolor=transparent,    
        font=dict(color=colour_white),
        margin=dict(t=20, b=20, l=20, r=20),
        height=470
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
        title = "Total Unique Starters Used"
    elif filter_type in ("GKP", "DEF", "MID", "FWD"):
        agg = df[df.position == filter_type].groupby("User")["PlayerID"].nunique()
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