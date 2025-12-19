# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yahooquery as yf
import math

import streamlit as st
import plotly.graph_objs as go

# Initiate dashboard
st.set_page_config(page_title="Daily Change Dashboard", layout="wide")

# Functions

# Sidebar
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Search Ticker", value="SPY")
start_date = st.sidebar.text_input("Start Date", value="2025-01-01")

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)
if ticker:
    hist = yf.Ticker(ticker).history(start=start_date).reset_index()

    daily_diffs_list = []
    for i in range(1, hist.shape[0]):
        p1 = hist['close'].iloc[i-1]
        p2 = hist['close'].iloc[i]
        diff = round(((p2 - p1) / p2) * 100, 5)
        date = hist['date'].iloc[i]
        daily_diffs_list.append([date, diff])
    daily_diffs = pd.DataFrame(data=daily_diffs_list, columns=[
        "date", "daily_change"])
    stats = daily_diffs.describe()
    std = stats.loc['std', 'daily_change']
    mean = stats.loc['mean', 'daily_change']

    with c1:
        st.subheader(f"{ticker} Price History (Using plotly.graph_objs)")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist['date'],
                y=hist['close'],
                mode="lines",
                name="Close"
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, width="stretch")

    with c4:
        st.subheader(f"{ticker} Price History (Using matplotlib.pyplot)")

        bins = np.array(range(-5, 6, 1))
        table = np.zeros((len(bins), len(bins)))

        for i in range(len(daily_diffs_list) - 1):
            today = 0
            for j in range(len(bins)):
                today = j
                if daily_diffs_list[i][1] <= bins[j]:
                    break

            tomorrow = 0
            for j in range(1, len(bins)):
                tomorrow = j
                if daily_diffs_list[i+1][1] <= bins[j]:
                    break

            table[today][tomorrow] += 1

        change_counts_df = pd.DataFrame(table, columns=bins, index=bins)

        change_counts_df.style.background_gradient(cmap='Blues', axis=None)
        st.dataframe(
            change_counts_df,
            width="stretch",
            height=430
        )

    with c3:
        st.subheader("Daily Directional Changes")

        x = daily_diffs['date']
        y = daily_diffs['daily_change']

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

        thresh = 0.5

        for i in range(len(y) - 1):
            # green day followed by a red day
            if y[i] >= thresh and y[i+1] <= -thresh:
                ax1.scatter(x.iloc[i], y[i], color='blue')
                ax1.scatter(x.iloc[i+1], y[i+1], color='black')
                ax1.plot([x.iloc[i], x.iloc[i+1]], [y[i], y[i+1]], color='red')

            # red day followed by a green day
            elif y[i] <= -thresh and y[i+1] >= thresh:
                ax2.scatter(x.iloc[i], y[i], color='blue')
                ax2.scatter(x.iloc[i+1], y[i+1], color='black')
                ax2.plot([x.iloc[i], x.iloc[i+1]],
                         [y[i], y[i+1]], color='green')

        # Daily price change standard deviations
        for ax in [ax1, ax2]:
            ax.axhline(y=2*std, color='b', linestyle='--',
                       label=f'+2 std ({round(2*std, 3)})')
            ax.axhline(y=1*std, color='b', linestyle='dotted',
                       label=f'+1 std ({round(1*std, 3)})')
            ax.axhline(y=mean, color='g', linestyle='dotted',
                       label=f'mean ({round(mean, 3)})')
            ax.axhline(y=-1*std, color='r', linestyle='dotted',
                       label=f'-1 std ({round(-1*std, 3)})')
            ax.axhline(y=-2*std, color='r', linestyle='--',
                       label=f'-2 std ({round(-2*std, 3)})')

            ax.set_xlabel('Date')
            ax.set_ylabel('Daily Percent Change')
            ax.legend(bbox_to_anchor=(1.02, 0.5),
                      loc='center left', borderaxespad=0.)

        ax1.set_title(
            f'{ticker} Daily Change: Green day (>={thresh}%) follwed by a red day (<={thresh}%)')
        ax2.set_title(
            f'{ticker} Daily Change: Red day (<={thresh}%) follwed by a green day(>={thresh}%)')

        st.pyplot(fig)

    with c2:
        st.subheader("Strategy Comparisons")

        def buy(bal, price):
            shares = int((bal//price))
            cost = shares * price
            return shares, cost

        # Buy Close Sell Open
        def bcso(balance, hist):
            total_pnl = 0
            running_pnl = []

            for i in range(hist.shape[0] - 1):
                close = hist['close'].iloc[i]
                shares, cost = buy(balance, close)
                pos_pnl = shares*hist['open'].iloc[i+1] - cost
                balance += pos_pnl
                total_pnl += pos_pnl
                running_pnl.append(total_pnl)

            return total_pnl, running_pnl

        # Buy Open Sell Close
        def bosc(balance, hist):
            total_pnl = 0
            running_pnl = []

            for i in range(hist.shape[0]):
                open = hist['open'].iloc[i]
                shares, cost = buy(balance, open)
                pos_pnl = shares*hist['close'].iloc[i] - cost
                balance += pos_pnl
                total_pnl += pos_pnl
                running_pnl.append(total_pnl)

            return total_pnl, running_pnl

        # Buy and Hold
        def buy_and_hold(balance, hist):
            total_pnl = 0
            running_pnl = []

            buy_open = hist['open'].iloc[0]
            shares, cost = buy(balance, buy_open)

            for i in range(hist.shape[0]):
                close = hist['close'].iloc[i]
                total_pnl = shares*close - cost
                running_pnl.append(total_pnl)

            return total_pnl, running_pnl

        bcso_pnl, bcso_run_pnl = bcso(10000, hist)
        bosc_pnl, bosc_run_pnl = bosc(10000, hist)
        bah_pnl, bah_run_pnl = buy_and_hold(10000, hist)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist['date'],
                y=bcso_run_pnl,
                mode="lines",
                name=f'BCSO PnL: ${round(bcso_pnl, 2)}',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=hist['date'],
                y=bosc_run_pnl,
                mode="lines",
                name=f'BAH PnL: ${round(bah_pnl, 2)}',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=hist['date'],
                y=bah_run_pnl,
                mode="lines",
                name=f'BOSC PnL: ${round(bosc_pnl, 2)}',
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, width="stretch")
