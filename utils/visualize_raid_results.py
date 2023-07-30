import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", help="path to the results file", default="../assets/season_7_results.csv")
    parser.add_argument("--member_file", help="path to the member file", default="../assets/season_7_members.csv")
    parser.add_argument("--chart_studio", help="send the plot to chart studio", action="store_true")
    parser.add_argument("--chart_studio_username", help="chart studio username", default="")
    parser.add_argument("--chart_studio_api_key", help="chart studio api key", default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # columns are Boss + lvl, member, damage and K (boolean)
    df = pd.read_csv(args.results_file)
    # split the boss and level into two columns based on "LvL"
    df[["Boss", "Level"]] = df["Boss"].str.split("LvL", expand=True)
    # remove the LvL from the level column
    df["Level"] = df["Level"].str.replace("LvL", "")
    # rename the notes column to Kill
    df = df.rename(columns={"Notes": "Kill"})
    # set K to true if the member killed the boss
    df["Kill"] = df["Kill"].apply(lambda x: True if x == "K" else False)
    # convert the damage to millions
    df["Damage (M)"] = df["Damage"].apply(lambda x: x / 1000000)
    # remove the damage column
    df = df.drop(columns=["Damage"])
    # convert the day to an integer
    df["Day"] = df["Day"].apply(lambda x: int(x.replace("Day", "")))

    # columns are Member and Synchro LvL
    df_members = pd.read_csv(args.member_file)
    # merge the two dataframes on the Member column
    df = pd.merge(df, df_members, on="Member")

    # plot the damage vs level and color by boss
    fig = go.Figure()
    for boss in df["Boss"].unique():
        df_boss = df[df["Boss"] == boss]
        # add member, boss level, synchro lvl, damage and kill to the hover text
        # boss is already in the legend and hover text
        df_boss["hover_text"] = df_boss["Member"] + "<br>" + \
                                "Boss Level: " + df_boss["Level"].astype(str) + "<br>" + \
                                "Synchro Level: " + df_boss["Synchro LvL"].astype(str) + "<br>" + \
                                "Damage: " + df_boss["Damage (M)"].astype(str) + "<br>" + \
                                "Day: " + df_boss["Day"].astype(str) + "<br>" + \
                                "Kill: " + df_boss["Kill"].astype(str)
        fig.add_trace(go.Scatter(x=df_boss["Synchro LvL"], y=df_boss["Damage (M)"], mode="markers", 
                                 hovertemplate=df_boss["hover_text"], name=boss))

    trendline_colors = ["red", "blue", "green", "orange", "purple"]
    for day in df["Day"].unique():
        df_day = df[df["Day"] == day]
        # fit a linear regression model
        model = LinearRegression()
        model.fit(df_day["Synchro LvL"].values.reshape(-1, 1), df_day["Damage (M)"].values.reshape(-1, 1))
        # plot the trendline
        fig.add_trace(
            go.Scatter(
                x=df_day["Synchro LvL"],
                y=model.predict(df_day["Synchro LvL"].values.reshape(-1, 1)).flatten(),
                mode="lines",
                name=f"Day {day} Trendline",
                # make transparent
                opacity=0.5,
                line=dict(color=trendline_colors[day - 1], width=2),
            )
        )
        # plot the equation of the trendline
        fig.add_annotation(
            x=400,
            y=1300 + (day * 50),
            text=f"Day {day} Trendline: {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}",
            showarrow=False,
            font=dict(size=12, color=trendline_colors[day - 1]),
        )
    fig.update_layout(title="Damage vs Synchro Level", xaxis_title="Level", yaxis_title="Damage (M)")
    
    # make a bar chart for each individual member with subbars for each hit against a boss
    # order by the x axis based on the Synchro LvL
    df_sum = df.groupby('Member')['Damage (M)'].sum().reset_index()
    fig2 = px.bar(df, x='Member', y='Damage (M)', color='Boss', orientation='v', color_continuous_scale=px.colors.qualitative.Plotly,
        hover_data={"Boss": True, "Damage (M)": ":.0f", "Level": True, "Kill": True, "Day": True, "Synchro LvL": True, "Member": False})
    # add the df_sum as a annotation on top of each bar
    total_damage_labels = [f"{row['Damage (M)']:.0f}" for index, row in df_sum.iterrows()]
    fig2.update_layout(annotations=[go.layout.Annotation(x=member, y=int(label)+100, text=label, showarrow=False) for \
        member, label in zip(df_sum['Member'], total_damage_labels)])
    fig2.update_xaxes(categoryorder='array', categoryarray=df_members.sort_values(by="Synchro LvL")["Member"])
    fig2.update_layout(title="Damage vs Member", xaxis_title="Member (Ordered by Synchro Level)", yaxis_title="Damage (M)", legend_title="Boss")
    fig2.update_layout(barmode='stack')
    fig2.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig2.update_traces(textposition='outside')

    if args.chart_studio:
        import chart_studio
        chart_studio.tools.set_credentials_file(username=args.chart_studio_username, api_key=args.chart_studio_api_key)
        chart_studio.tools.set_config_file(world_readable=True, sharing='public')
        
        # send the plot to chart studio
        chart_studio.plotly.plot(fig, filename="Damage vs Synchro Level", auto_open=True)
        chart_studio.plotly.plot(fig2, filename="Damage vs Member", auto_open=True)
    else:
        fig.show()
        fig2.show()