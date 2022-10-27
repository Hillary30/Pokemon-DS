import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import classification_report

#Load the databases
pokemon = pd.read_csv("pokemon.csv")  # Pokemon Dataset
combats = pd.read_csv("combats.csv")  # Combats Dataset

#Print first 5 of pokemon datasets
print(pokemon.head())
print(combats.head())
print(pokemon.nunique())

# Plot the number of pokemon present in each category of "type 1"
ax = pokemon['Type 1'].value_counts().plot(kind='bar',
    figsize=(14,8),
    title="Number of pokemons based on their type 1")
ax.set_xlabel("Pokemon Type 1")
ax.set_ylabel("Frequency")

# Plot the number of pokemon present in each category of "type 2"
ax = pokemon['Type 2'].value_counts().plot(kind='bar',
    figsize=(14,8),
    title="Number of pokemons based on their type 2")
ax.set_xlabel("Pokemon Type 2")
ax.set_ylabel("Frequency")



# Plot the number of pokemon present in each generation.
generation =  dict(pokemon['Generation'].value_counts())
gen_counts = generation.values() # No of pokemon in each generation
gen = generation.keys()  # Type of generation

fig = plt.figure(figsize=(8, 6))
fig.suptitle("Percentage of generation based distribution of pokemon")
ax = fig.add_axes([0,0,1,1])
explode = (0.1, 0, 0, 0, 0, 0)  # explode 1st slice
ax.axis('equal')

plt.pie(gen_counts, labels = gen,autopct='%1.2f%%', shadow=True, explode=explode)
plt.show()


# Plot the number of legendary and non-legendary pokemon
generation =  dict(pokemon['Legendary'].value_counts())
gen_counts = generation.values() 
gen = generation.keys()

fig = plt.figure(figsize=(8, 6))
fig.suptitle("Percentage of lengendary pokemon in dataset (False: Not Lengendary, True: Legendary)")
ax = fig.add_axes([0,0,1,1])
explode = (0.2, 0)  # explode 1st slice
ax.axis('equal')

plt.pie(gen_counts, labels = gen,autopct='%1.2f%%', shadow=True, explode=explode)
plt.show()



#DATA PROCESSING
#Handle missing data
pokemon["Type 2"] = pokemon["Type 2"].fillna("NA")

# Convert "Legendary" column, False is converted to 0 and True is converted to 1.
pokemon["Legendary"] = pokemon["Legendary"].astype(int)

#Convert to dataframe
#FeatureHasher converts the columns Type1/Type2 into numbers
h1 = FeatureHasher(n_features=5, input_type='string')
h2 = FeatureHasher(n_features=5, input_type='string')
d1 = h1.fit_transform(pokemon["Type 1"])
d2 = h2.fit_transform(pokemon["Type 2"])


# Convert to dataframe of Type1 and Type2
d1 = pd.DataFrame(data=d1.toarray())
d2 = pd.DataFrame(data=d2.toarray())


# Drop Type 1 and Type 2 column from Pokemon dataset 
# #and concatenate the above two dataframes. = new data set
pokemon = pokemon.drop(columns = ["Type 1", "Type 2"])
pokemon = pd.concat([pokemon, d1, d2], axis=1)


x = pokemon.loc[pokemon["#"]==266].values[:, 2:][0]
print(x)
y = pokemon.loc[pokemon["#"]==298].values[:, 2:][0]
print(y)
z = np.concatenate((x,y))
#z


data = []
for t in combats.itertuples():
    first_pokemon = t[1]
    second_pokemon = t[2]
    winner = t[3]
    
    x = pokemon.loc[pokemon["#"]==first_pokemon].values[:, 2:][0]
    y = pokemon.loc[pokemon["#"]==second_pokemon].values[:, 2:][0]
    diff = (x-y)[:6]
    z = np.concatenate((x,y))
    
    if winner == first_pokemon:
        z = np.append(z, [0])
    else:
        z = np.append(z, [1])
        
    data.append(z)
data = np.asarray(data) 


# train-test split procedure is used to estimate the performance of machine learning
# algorithms when they are used to make predictions on data not used to train the model.
X = data[:, :-1].astype(int)
y = data[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#RandomForestClassifier = many decision trees
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(X_train,y_train)
pred = model.predict(X_test)
print('Accuracy of {}:'.format(data), accuracy_score(pred, y_test))

print('Accuracy :', accuracy_score(pred, y_test))
print(classification_report(y_test, pred))