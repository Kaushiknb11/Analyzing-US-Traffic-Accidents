# Importing required libraries 

# Data Mining
import pandas as pd
import seaborn as sns
import numpy as np
# visualisation  
import matplotlib.pyplot as plt
import missingno as msno
import plotly.graph_objects as go
import matplotlib as mpl
from plotly.offline import iplot
import circlify
from wordcloud import WordCloud, STOPWORDS
import folium

# Math 
import math

# Reading Data
df1 = pd.read_csv('cleaned_US_Accidents_Dec21_updated.csv')

# Choropleth Plot
print("Traffic Accidents by State")
state_counts = df1["State"].value_counts()
fig = go.Figure(data=go.Choropleth(locations=state_counts.index, z=state_counts.values.astype(float), locationmode="USA-states", colorscale="turbo"))
fig.update_layout(title_text="Traffic Accidents by State", geo_scope="usa")
fig.show()
iplot(fig, image='png')


# Circle Plot
print("Top US cities with most Accidents")
circles = circlify.circlify(cities[:15].tolist(), 
                            show_enclosure=False, 
                            target_enclosure=circlify.Circle(x=0, y=0)
                           )
circles.reverse()
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
ax.axis('off')
lim = max(max(abs(circle.x)+circle.r, abs(circle.y)+circle.r,) for circle in circles)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

# print circles
for circle, label, emi, color in zip(circles, cities[:15].index, cities[:15], list(sns.color_palette(palette="pastel", n_colors=len(df)).as_hex())):
    x, y, r = circle
    ax.add_patch(plt.Circle((x, y), r, alpha=0.9, color = color))
    plt.annotate(label +'\n'+ format(emi, ","), (x,y), size=12, va='center', ha='center')
plt.xticks([])
plt.yticks([])
plt.title("Top US cities with most Accidents")
plt.show()


# Histogram
print('Accident by Hour')
axis=sns.displot(df1.Start_Time.dt.hour, bins = 24).set(title='Accidents by Hour');
axis.set(xlabel='Start_Time(hr)', ylabel='Count_of_Accidents')
plt.show()

# Donut Chart
print("Donut Chart Accidents by Weekday")
my_circle = plt.Circle((0, 0), 0.7, color='white')
names = ['Fri', 'Thur', 'Wed', 'Tue','Mon','Sat','Sun']
plt.pie(xx['Start_Time'], labels=names, autopct='%1.1f%%',
        colors=['red', 'green', 'blue', 'yellow', 'grey', 'orange', 'purple' ])
  
p = plt.gcf()
p.gca().add_artist(my_circle)
  
# Show the graph
plt.title("Accidents by Weekday")
plt.show()


# Radar Plot
print("Radar Plot of Accidents by Weather")
plt.gcf().set_size_inches(6, 6)
sns.set_style('darkgrid')

#set max value
max_val = max(Weather_Condition)*1.01
ax = plt.subplot(projection='polar')

#set the subplot 
ax.set_theta_zero_location('N')
ax.set_theta_direction(1)
ax.set_rlabel_position(0)
ax.set_thetagrids([], labels=[])
ax.set_rgrids(range(len(Weather_Condition)), labels= ['Clear', 'Cloudy', 'Rain', 'Fog','Snow','Windy','Thunderstorm','Smoke','Sand','Hail','Tornado'])

#set the projection
ax = plt.subplot(projection='polar')

for i in range(len(Weather_Condition)):
    ax.barh(i, list(Weather_Condition)[i]*2*np.pi/max_val,
            label=list(['Clear', 'Cloudy', 'Rain', 'Fog','Snow','Windy','Thunderstorm','Smoke','Sand','Hail','Tornado'])[i], color=['red', 'green', 'blue', 'yellow', 'grey', 'orange', 'purple', 'violet','magenta','cyan','white'][i])
    
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title("Accidents by Weather")
plt.show()  

# Top Street dataframe
streets = df1.Street.value_counts()
top_streets = streets.head(10)

# Bar Plot 

print('Bar Plot of No. of accidents')
plt.figure(figsize=(10,5))
sns.barplot(y=top_streets.index, x=top_streets)
plt.title("Top 10 Accident Prone Streets in US ",size=10)
plt.xlabel('Streets')
plt.ylabel('No.of accidents');
plt.show()

# Line Chart
Print('Temporal Plot of Line Chart')
plt.figure(figsize=(10,5))
sns.lineplot(data = df1.Start_Time.dt.year.value_counts())
plt.title(" Accident cases yearly (2016-2021) in US ",size=12)
plt.xlabel('Year')
plt.ylabel('No. of accidents');
plt.show()

# Quick Pie
plt.show(df1.groupby('Severity').size().plot(kind='pie', autopct='%.2f', title='Accidents by Severity'))

# Word Cloud
word_sample=df1.sample(n=20000, random_state=1)
print(word_sample['Severity'].value_counts())
print('\n')
print('It looks like we should have enough text data for a Word Cloud of Severity levels 2, 3, and 4.')

word_sample_S4=word_sample[word_sample['Severity']==4] #Making subsets for us to use in our wordcloud generation 
word_sample_S3=word_sample[word_sample['Severity']==3]
word_sample_S2=word_sample[word_sample['Severity']==2]
word_sample_S1=word_sample[word_sample['Severity']==1]

#Custome Color Map for Severity 1
cmap_O = mpl.cm.Oranges(np.linspace(0,1,20))
cmap_O = mpl.colors.ListedColormap(cmap_O[10:,:-1])

#Custome Color Map for Severity 2
cmap_O = mpl.cm.Oranges(np.linspace(0,1,20))
cmap_O = mpl.colors.ListedColormap(cmap_O[10:,:-1])

#Custome Color Map for Severity 3
cmap_R = mpl.cm.Reds(np.linspace(0,1,20))
cmap_R = mpl.colors.ListedColormap(cmap_R[10:,:-1])

#Custome Color Map for Severity 4
cmap_H = mpl.cm.hot_r(np.linspace(0,1,20))
cmap_H = mpl.colors.ListedColormap(cmap_H[10:,:-1])

#Font parameters for our Title
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 36,
        }

#Creating variable with data for this Word Cloud
textS2 = ' '.join(word_sample_S2['Description'].tolist())

#Creating Mask for Word Cloud
stop_words=set(STOPWORDS)

#Word Cloud Creation 
wc=WordCloud(width=400,height=200,random_state=101, max_font_size=450,
                 min_font_size=1,stopwords=stop_words,background_color="white",
                 scale=3,max_words=400,collocations=True,colormap=cmap_O)

#Generate Word Cloud
wc.generate(str(textS2))


#show
print('Word Cloud for Severity 2')
fig=plt.figure(figsize=(15,20))
#plt.ylim(-250,2700)
#plt.xlim(0,2650)
plt.gca().invert_yaxis()
#plt.gca().invert_xaxis()
plt.axis("off")
plt.title('Accidents: Severity 2',fontdict=font)
plt.imshow(wc,interpolation='bilinear')
plt.show()


print('Word Cloud for Severity 3')
#Creating variable with data for this Word Cloud
textS3 = ' '.join(word_sample_S3['Description'].tolist())

#Creating Mask for Word Cloud
stop_words=set(STOPWORDS)

#Word Cloud Creation 
wc=WordCloud(width=400,height=200,random_state=101, max_font_size=450,
                 min_font_size=1.5,stopwords=stop_words,background_color="white",
                 scale=3,max_words=400,collocations=True,colormap=cmap_R)

#Generate Word Cloud
wc.generate(str(textS3))


#Show
fig=plt.figure(figsize=(30,10))
#plt.ylim(-300,2600)
#plt.xlim(0,7500)
#plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
plt.axis("off")
plt.title('Accidents: Severity 3',fontdict=font)
plt.imshow(wc,interpolation='bilinear')
plt.show()


#Creating variable with data for this Word Cloud
print('Word Cloud for Severity 4')
textS4 = ' '.join(word_sample_S4['Description'].tolist())

#Creating Mask for Word Cloud
stop_words=set(STOPWORDS)

#Word Cloud Creation 
wc=WordCloud(width=400,height=200,random_state=101, max_font_size=450,
                 min_font_size=1.5,stopwords=stop_words,background_color="white",
                 scale=3,max_words=400,collocations=True,colormap=cmap_H)

#Generate Word Cloud
wc.generate(str(textS4))


#Show
fig=plt.figure(figsize=(30,10))
#plt.ylim(500,1300)
plt.gca().invert_yaxis()
plt.axis("off")
plt.title('Accidents: Severity 4',fontdict=font)
plt.imshow(wc,interpolation='bilinear')
plt.show()


# Countplot 
s=sns.countplot(x='Month',hue='Severity',data=df1,palette='mako_r').set_title( 'Accidents by Month')
s.show()

# Folium Map
#First we will subset our data down to just the state level
df_CT=df.loc[df['State']=='CA'].copy()
#County level 
df_VT=df_CT.loc[df_CT['County']=='Ventura'].copy()
print('The new shape after defining the scope:',df_VT.shape,',that is something i can work with')
pd.set_option('display.max_columns', None)
df_VT.head(1)

#Randomly subsetting the data to make it more manageable 
df_VT_half=df_VT.sample(frac=0.5, replace=True, random_state=101)
df_VT_half.head(5)


#Creating map of our locaiton of choice Fairfield County
VT_map = folium.Map(location=[34.27,-119.22],tiles = 'Stamen Terrain', zoom_start=10)
VT_map
from folium import plugins
from folium.plugins import HeatMap

#Making sure our data is in the correct type
df_VT_half['Start_Lat']=df_VT_half['Start_Lat'].astype(float)
df_VT_half['Start_Lng']=df_VT_half['Start_Lng'].astype(float)

#Subsetting data for visualization
df_VTHeat=df_VT_half[['Start_Lat','Start_Lng']]
df_VTHeat=df_VTHeat.dropna(axis=0,subset=['Start_Lat','Start_Lng'])

#Creating and Attaching heatmap to our map
VTHeat_data=[[row['Start_Lat'],row['Start_Lng']] for index, row in df_VTHeat.iterrows()]
HeatMap(VTHeat_data,blur=10,radius=15,gradient={0.4: 'green', 0.65: 'yellow', 1: 'red'}).add_to(VT_map)

#show
VT_map

# Plotting Interstates on Choropleth
df2['interstateplus'] = df2["Description"].str.contains('INTERSTATE| interstate|I-2| I-4 |I-5 | I-8| I-10 |I-11| I-12| I-14 |I-15| I-16| I-17 |I-19| I-20| I-22 |I-24|I-25| I-26| I-27 |I-29| I-30| I-31 |I-35| I-37| I-39 |I-40 | I-41| I-42 |I-43| I-44| I-45 |I-49| I-55| I-57 |I-59 | I-64| I-65 |I-66| I-68| I-69 |I-70| I-71| I-72 |I-73| I-74| I-75 |I-75| I-76| I-77 | I-78| I-79 |I-80|I-81 |I-82 |I-83 |I-84 |I-85 |I-86 |I-87 |I-88 | I-89| I-90 | I-90N| I-91 |I-93 |I-94 |I-95| I-96| I-97 | I-99') 

df2['rtstreet'] = df2["Street"].str.contains('INTERSTATE| interstate|I-2| I-4 |I-5 | I-8| I-10 |I-11| I-12| I-14 |I-15| I-16| I-17 |I-19| I-20| I-22 |I-24|I-25| I-26| I-27 |I-29| I-30| I-31 |I-35| I-37| I-39 |I-40 | I-41| I-42 |I-43| I-44| I-45 |I-49| I-55| I-57 |I-59 | I-64| I-65 |I-66| I-68| I-69 |I-70| I-71| I-72 |I-73| I-74| I-75 |I-75| I-76| I-77 | I-78| I-79 |I-80|I-81 |I-82 |I-83 |I-84 |I-85 |I-86 |I-87 |I-88 | I-89| I-90 | I-90N| I-91 |I-93 |I-94 |I-95| I-96| I-97 | I-99', na=False) 

df2["InstHwy"] = df2[['interstateplus', 'rtstreet']].any(axis='columns')

print('Accidents on Interstate Highways') # Set the figure title
plotdata = [df2[~df2['InstHwy']][['Start_Lng','Start_Lat']], df2[df2['InstHwy']][['Start_Lng','Start_Lat']]]
markers = ["o","+"]
fcolors = ["none", "red"]
ecolors = ["lightblue", "none"]
plotlabels = ['Interstate', "Not_interstate"]
plt.figure(figsize=(18, 13)) # Specify figure size
for data, label, marker, fcolor, ecolor in zip(plotdata, plotlabels, markers, fcolors, ecolors):
    plt.scatter(x=data['Start_Lng'],y=data['Start_Lat'],marker=marker,facecolors=fcolor, edgecolors=ecolor)
plt.xlabel('Start_Long') # Set the label for the x axis
plt.ylabel('Start Lat') # Set the label for the y axis
plt.title('Accidents on Interstate Highways') # Set the figure title
plt.show()


# Violin Plots
# Violin Plots (this is a check to see if anything interesting is here)
# and determine if there is a real difference

print('Severity of Accident by Road Type') 
plotlabels = ['Interstate', "Not_interstate"]
plotdata = [df2[df2['Road_type']==label]['Severity'] for label in plotlabels]
colors = ['green','magenta']
plt.figure(figsize=(7.5, 7.5))
vparts = plt.violinplot(plotdata, showextrema=False) # Violin plot including density estimation
for patch, color in zip(vparts['bodies'], colors):
    patch.set_color(color) # Set color for violin plot
plt.boxplot(plotdata, widths=0.15, zorder=2, labels=plotlabels) # Overlay box plot
plt.xlabel('Type of Road') # Set x label
plt.ylabel('Severity') # Set y label
plt.ylim(0, 5) # Set limits for x axis
plt.title('Severity of Accident by Road Type') # Set title 
plt.grid(zorder=0) # Add grid
plt.savefig("voilplroadtype.png", format="png")
plt.show()


# Violin Plots
print('Distance From Accident On Interstate') # Set title 
plotlabels = ['Interstate', "Not_interstate"]
plotdata = [df2[df2['Road_type']==label]['Distance(mi)'] for label in plotlabels]
colors = ['green','magenta']
plt.figure(figsize=(7.5, 7.5))
vparts = plt.violinplot(plotdata, showextrema=False) # Violin plot including density estimation
for patch, color in zip(vparts['bodies'], colors):
    patch.set_color(color) # Set color for violin plot
plt.boxplot(plotdata, widths=0.15, zorder=2, labels=plotlabels) # Overlay box plot
plt.xlabel('Type of Road') # Set x label
plt.ylabel('Distance (Miles)') # Set y label
plt.ylim(0, 5) # Set limits for x axis
plt.title('Distance From Accident On Interstate') # Set title 
plt.grid(zorder=0) # Add grid
plt.savefig("violindist.png", format="png")
plt.show

new=df2.groupby(['State','City','County']).count()['Severity']
new=new.reset_index()
print('Total Accidents')
n_df=new.sample(n=2000)
n_df.rename(columns={"Severity": "Total_Accidents"}, inplace=True)
n_df["US"] = "US" # in order to have a single root node
fig = px.treemap(n_df, path=['US','State','City', 'County'], values='Total_Accidents', width = 2000, height = 1000, color = "Total_Accidents", color_continuous_scale=["#32a852", "#3261a8", "#a83259"])
fig.update_traces(root_color="lightgrey")
#fig.update_layout(margin = dict(t=50, l=25, r=25))
fig.show()
iplot(fig,image= "png")

###########################################################################################
