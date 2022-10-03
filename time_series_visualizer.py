import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#----------------------------------------------------#
#----------------------------------------------------#
import numpy as np
#----------------------------------------------------#
# CALENDAR MONTHS (FULL/ABBREVIATED NAMES):
import calendar

month_full_names = dict(enumerate(calendar.month_name))

month_abbr_names = dict(enumerate(calendar.month_abbr))
monthAbbrSeries = pd.Series(np.arange(1,13)).map(month_abbr_names)
monthAbbrArray = np.array(monthAbbrSeries.to_list())
#----------------------------------------------------#
#----------------------------------------------------#

# Import data (Make sure to parse dates. Consider setting index column to 'date'.)
df = pd.read_csv(
    'fcc-forum-pageviews.csv',
    header=0, # first row has headers
    index_col='date', #=[0],
    parse_dates=['date'], #=[0],
    sep=",")

# Clean data
#Interval Notation: [2.5%, 97.5%]
mask1 = (df['value'] >= df['value'].quantile(.025))
mask2 = (df['value'] <= df['value'].quantile(1-.025))
maskTotal = mask1 & mask2
df = df[maskTotal]



def draw_line_plot():
    # copy dataframe
    df_line = df.copy()
  
    # Draw line plot  
    plt.figure(1)
    
    axes = df_line.plot(
        kind='line',
        figsize=(17,7),
        title="Daily freeCodeCamp Forum Page Views 5/2016-12/2019",
        xlabel="Date", ylabel='Page Views',
        color="crimson", linestyle="-")
    
    # align x labels horizontally and center to x ticks
    plt.xticks(rotation="horizontal", horizontalalignment='center')

    # hide legend
    axes.get_legend().set_visible(False)

    # assign figure
    fig = axes.figure

    # Save image and return fig (don't change this part)
    fig.savefig('line_plot.png')
    return fig



def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar1238 = df.copy()

    # create a column in "YYYY-MM" string format (as to be able to sort it out by year and month)
    df_bar1238['year_month'] = df_bar1238.index.strftime("%Y-%m")
    df_bar1238['year'] = df_bar1238.index.year
    df_bar1238['month'] = df_bar1238.index.month

    # group dataframe by year and month (as to get into a shape of (44,)),
    # and get the sum ('value' column) of views per month (result of 'value' column)
    df_bar44 = df_bar1238.groupby(by=['year_month', 'year', 'month'], as_index=False)['value'].sum()
    
    # calculate average daily views per month (shape of (44,))
    days_per_month = df_bar1238['year_month'].value_counts().sort_index(ascending=True)
    views_per_month = df_bar44['value'].values
    avg_daily_views_per_month = (views_per_month / days_per_month)
    
    # drop 'year_month' and 'value' columns
    df_bar44 = df_bar44.drop(columns=['year_month', 'value'])
    
    # create 'average_views' column (shape of (44,))
    df_bar44['average_views'] = avg_daily_views_per_month.values

    # reformat dataframe into wide format (index in Years, single-column in Months, values in Average Views Per Month)
    df_bar = pd.pivot(df_bar44, index='year', columns='month', values='average_views')
    
    # get months names based on their numbers (1-12)
    df_bar.columns = df_bar.columns.map(month_full_names)
    
    # fill NaN values (Jan, Feb, Mar, and Apr for 2016) with zeros (0's)
    df_bar.fillna(0, inplace=True)
    
  
    # Draw bar plot
    plt.figure(2)

    axes = df_bar.plot(kind='bar', figsize=(10,10))
    
    axes.legend(title='Months', loc='upper left',
                fontsize='x-large',
                title_fontsize='x-large')
    
    # resize x and y labels
    plt.xlabel(xlabel='Years', fontsize='xx-large')
    plt.ylabel(ylabel='Average Page Views', fontsize='xx-large')
    
    # resize x and y ticks
    plt.xticks(fontsize='x-large', rotation=90, ha='center')
    plt.yticks(fontsize='x-large')

    # assign figure
    fig = axes.figure
  
    # Save image and return fig (don't change this part)
    fig.savefig('bar_plot.png')
    return fig



def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df_box = df.copy()
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box.date]
    df_box['month'] = [d.strftime('%b') for d in df_box.date]

    # Draw box plots (using Seaborn)
    plt.figure(5)
    
    # create the subplots figure and axes
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(22,7))
    
    
    # create first boxplot
    g1 = sns.boxplot(data=df_box, x='year', y='value', ax=axs[0],
                width=0.8, linewidth=0.7, fliersize=2)
    
    # set title
    axs[0].set_title('Year-wise Box Plot (Trend)', fontsize='x-large')
    
    # set x label, ticks and tick labels
    x_domain = df_box['year'].unique()
    axs[0].set_xlabel(xlabel='Year', fontsize='large')
    axs[0].set_xticks(ticks=np.array(range(len(x_domain))))
    axs[0].set_xticklabels(labels=x_domain, fontsize='large')
    
    # set y label, ticks and tick labels
    # (in this case, y axis values will repeat for both subplots)
    y_ticks = axs[0].get_yticks().astype('int')
    axs[0].set_ylabel(ylabel='Page Views', fontsize='large')
    axs[0].set_yticks(ticks=y_ticks) # sets visually on plot y ticks between values 0 and 200_000
    axs[0].set_yticklabels(labels=y_ticks, fontsize='large')
    
    
    # create second boxplot
    g2 = sns.boxplot(data=df_box,
                    x='month', y='value', ax=axs[1],
                    width=0.8, linewidth=0.7, fliersize=2,
                    order=monthAbbrArray)
    
    # set title
    axs[1].set_title('Month-wise Box Plot (Seasonality)', fontsize='x-large')
    
    # set x label, ticks and tick labels
    x_domain = monthAbbrArray
    axs[1].set_xlabel(xlabel='Month', fontsize='large')
    axs[1].set_xticks(ticks=np.array(range(len(x_domain))))
    axs[1].set_xticklabels(labels=x_domain, fontsize='large')
    
    # set y label, ticks and tick labels
    #y_ticks = (the same as the first boxplot)
    axs[1].set_ylabel(ylabel='Page Views', fontsize='large')
    axs[1].set_yticks(ticks=y_ticks) # sets visually on plot y ticks between values 0 and 200_000
    axs[1].set_yticklabels(labels=y_ticks, fontsize='large')

    # assign figure
    fig = f

    # Save image and return fig (don't change this part)
    fig.savefig('box_plot.png')
    return fig
