#!/usr/bin/env python
# coding: utf-8

# # 🔥 Konkurs & projekt do Twojego portfolio 
# 
# 
# ### Celem jest poświęcić więcej czasu na  samodzielną pracę, mieć w swoim portfolio prototyp modelu ML rozwiązujący bardzo konkretny problem i przy okazji dobrze się bawić :)  
# 
# #### 🕵️‍♀️ Kontekst wyzwania 
# 
# Opieka medyczna rozwija się, a przynajmniej próbuje i w związku z tym pojawiają się różne programy, które to napędzają. Teraz dotkniemy tematu leczenia przecukrzenia ([hiperglikemii](https://pl.wikipedia.org/wiki/Hiperglikemia)). Na przykład w Stanach Zjednoczonych jest taki program [Hospital Readmission Reduction Program (HRRP)](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/HRRP/Hospital-Readmission-Reduction-Program.html), dzięki któremu próbuje się minimalziować refundację dla szpitali o ponadprzeciętnej readmisji. Dla tych szpitali, które są obecnie karane w ramach tego programu, jednym z rozwiązań jest stworzenie interwencji w celu zapewnienia dodatkowej pomocy pacjentom o zwiększonym ryzyku readmisji. 
# 
# ❓ Pytanie jednak jest takie: jak rozpoznać takich pacjentów? No właśnie tutaj może przyjść z pomocą uczenie maszynowe. Tym zajmiemy się w tym konkursie.
# 
# ![](../images/competitions2019.png)
# 
# Zbadajmy kontekst, jak duży jest problem w Polsce (w Stanach jeszcze jest gorzej). 
# 
# 
# W Polsce ponad 3 miliony osób chorujących na cukrzycę (~9% populacji). 90% zachorowań dotyczy cukrzycy typu 2, której można zapobiegać! Szacuje się, że w 2035 r. na świecie liczba chorych na cukrzycę będzie wynosić około 592 miliony.😱
# 
# 
# 
# ## ❓ Czym jest cukrzyca?
# 
# Cukrzyca jest chorobą charakteryzującą się „podwyższonym stężeniem cukru (glukozy) we krwi”. Przyczyną przecukrzeń jest brak insuliny wynikający z uszkodzenia komórek beta trzustki – czytamy w „Narodowym Programie Przeciwdziałania Chorobom Cywilizacyjnym” opracowanym przez Ministerstwo Zdrowia.
# 
# 
# ## ❓ Na czym polega konkurs?
# 
# 
# Twoim celem jest przewidzieć, **czy pacjent z cukrzycą zostanie odesłany do szpitala w ciągu 30 dni**, czyli kolumna `readmitted`. 
# - 0 oznacza, że nie ->  osoba nie została odesłana do szpitala w ciągu 30 dni
# - 1 oznacza tak -> był odesłany do szpitala w ciągu 30 dni 
# 
# 
# ### ⭐ Metryka sukcesu
# 
# Naszą metryką będzie **F1.5**. To oznacza, że będzie bardziej czułe dla `recall`, niż `precision`.
# 
# ### Kaggle
# 
# Konkurs będzie dostępny na Kaggle - link do [konkursu](https://www.kaggle.com/t/0dcd1f5e99fa4cd98db2451e636de318).  
# 
# ### Rejestracja
# 
# Jeśli jeszcze nie masz konta na Kaggle, to proszę [załóż konto](https://www.kaggle.com/).
# 
# Kaggle to jest największa społeczność uczenia maszynowego na świecie (póki co DataWorkshop jest tylko w Polsce, ale to pewnie się zmieni ;)). Dlatego jeśli chcesz zająć się tym na poważnie, zainspirować się lub od czasu do czasu brać udział w konkursach na Kaggle, to polecam założyć tam konto ;)
# 
# ### Regulamin
# 
# 1. Konkurs rozpoczyna się **23 marca 2023 r** i kończy się w niedzielę, **2 kwietnia 2023 r. o 23:59** (Warszawa).
# 2. Osoba, która będzie na pierwszym miejscu dostaje nagrodę - Gravitrax, za drugie i trzecie miejsca będą kubki termiczne z logo DataWorkshop. Miejsca będą widoczne na Leader Board przez cały czas trwania konkursu.
# 3. Zbiór danych został podzielony na dwie części: publiczny i prywatny. Publiczny zbiór danych zawiera cechy + odpowiedź, prywatny zbiór tylko cechy. Należy zrobić predykcję na prywatnym zbiorze i wysłać na Kaggle. Gdy skończy się konkurs, to ostateczny wynik będzie na zbiorze prywatnym.
# 4. Dane są przygotowane w celach edukacyjnych, zakazane jest ich używanie w innym celu niż ten kurs oraz zakazane jest upublicznianie danych w dowolny sposób. Rozwiązanie ma być oparte tylko i wyłączenie przy użyciu uczenia maszynowego (trenowanie modelu na podstawie udostępnionych danych). Nie ma co kombinować, skup się na nauce :).
# 5. W tym konkursie nie można łączyć się w drużyny. Każdy uczestnik może dodać rozwiązanie (czyli zrobić tak zwany *submit*) do 5 prób rozwiązań w jeden dzień (dlatego warto zacząć wcześniej, będzie więcej prób).
# 6. Osoby, które zajmą 1, 2 oraz 3 miejsca opublikują wszystkim swoje rozwiązanie (w przeciągu tygodnia po wygranej) i udostępnią je podając link na Slacku.
# 7. Nagroda będzie wysłana do 2 tygodni na terenie Polski od przekazania nam danych do dostrczenia paczki. 
# 
# 
# ### Dane
# 
# W danych jest **66 221** wierszy, które zostały podzielone prawie na równe cześci:
# - train (**33 051** wierszy)
# - test (**33 170** wierszy)
# 
# Twoim zadaniem jest zrobić predykcje dla zbioru testowego.
# 
# 
# Jak robisz `submit` na Kaggle, to jest wyliczany wynik. Zwróć uwagę, że wynik jest wyliczany tylko na 30% danych testowych i ten wynik jest widoczny na tak zwanym "Public Leader Board". Gdy konkurs się skończy, wynik będzie przeliczony na wszystkich dostępnych danych testowych i będzie dostępny na tak zwanym "Private Leader Board". Dlaczego tak się robi? Chodzi o to, że Twoje rozwiązanie powinno być jak najbardziej stabilne na jak namniejszej jak i większej próbce danych. Dlatego musisz uważnie robić walidację, bo pierwsze miejsce na Public Leader Board nie musi być pierwszym na Private Leader Board :) Uważaj na to!
# 
# 
# ## Nagroda
# 
# ### Gravitrax - pierwsze miejsce
# ![Gravitrax](../images/gravitrax.jpg)
# 
# ### Kubek termiczny za 2 i 3 miejsce
# <div style="width: 700px; height: 500px; margin: 10px auto;">
#     <img src="../images/cup.png" style="height: 400px; float: left; " />
#     <img src="../images/cup.png" style="height: 400px; float: right;" />
# </div>

# In[1]:


import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.metrics import fbeta_score
from sklearn.dummy import DummyClassifier


# ## Dane
# To są prawdziwe dane.
# 
# **Uwaga!** Danych można używać tylko i wyłącznie w **celach edukacyjnych** (również nie można ich publikować lub dzielić się nimi z innymi)!

# In[2]:


train = pd.read_hdf('../input/diabetic_train.h5')
train.info()


# ## Opis danych
# 
# - **encounter_id** - Unikalny identyfikator spotkania.
# - **patient_nbr** - Unikalny identyfikator pacjenta
# - **race** - Rasa
# - **gender** - Płeć
# - **age** - Wiek pogrupowany w 10-letnich interwałach
# - **weight** - Waga w funtach
# - **admission_type_id** - Cyfrowy identyfikator rodzaju przyjęcia do szpitala (np. "awaryjny", "nowo narodzony" itp.)
# - **discharge_disposition_id** - Cyfrowy identyfikator rodzaju wypisania ze szpitala (np. "przeterminowane", "zwolniony do domu" itp.)
# - **admission_source_id** - Cyfrowy identyfikator źródła wizyty (np. "transfer z innego szpitala", "skierowanie od lekarza" itp.)
# - **time_in_hospital** - Czas w szpitalu w dniach
# - **payer_code** - Identyfikator rodzaju płatności (np. czy pacjent sam płacił albo czy miał ubezpieczenie w Blue Cross/Blue Shield itp.)
# - **medical_specialty** - Specjalizacja lekarza, który przyjął pacjenta
# - **num_lab_procedures** - Ilość testów laboratoryjnych przeprowadzonych w trakcie spotkania
# - **num_procedures** - Ilość procedur (innych, niż testy laboratoryjne) przeprowadzonych w trakcie spotkania
# - **num_medications** - Ilość unikalnych lekarstw podanych w trakcie spotkania
# - **number_outpatient** - Ilość wizyt ambulatorium przez pacjenta w ciągu roku poprzedzającego spotkanie
# - **number_emergency** - Ilość awaryjnych (ang. emergency) wizyt pacjenta w ciągu roku poprzedzającego spotkanie
# - **number_inpatient** - Ilość hospitalizowanych wizyt pacjenta w ciągu roku poprzedzającego spotkanie
# - **diag_1** - Pierwotna diagnoza (oznaczona jako 3 pierwsze cyfry wg standardu ICD9)
# - **diag_2** - Wtórna diagnoza (oznaczona jako 3 pierwsze cyfry wg standardu ICD9)
# - **diag_3** - Dodatkowa wtórna diagnoza (oznaczona jako 3 pierwsze cyfry wg standardu ICD9)
# - **number_diagnoses** - Ilość diagnoz wprowadzona do systemu
# - **max_glu_serum** - Wynik badań na glukozę.
# - **A1Cresult** - Wynik badania A1c. ">8" jeśli wynik był większy, niż 8%, ">7" jeśli wynik był większy, niż 7%, ale mniejszy, niż 8%. "normal" jeśli wynik jest mniejszy, niż 7%
# - **change** - Informacja, czy była zmiana w lekarstwach (zarówno dawka, jak i rodzaj leku)
# - **diabetesMed** - Informacja, czy była recepta na dowolne lekarstwa na cukrzycę
# - 24 kolumny z nazwami lekarstw (**metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, sitagliptin, insulin, glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, metformin-pioglitazone**) - Mówi o tym czy dawka na dane lekarstwo została zwiększona, zmniejszona, czy pozostała bez zmian, lub czy w ogóle nie było recepty na to lekarstwo
# - **readmitted** - Czy w ciągu 30 dni pacjent był ponownie skierowany do hospitalizacji
# - **id** - Unikalne id obserwacji (potrzebne do Kaggle)

# ## 🤖 Basic Model

# In[3]:


feats = ['encounter_id']
X = train[feats].values
y = train['readmitted'].values

model = DummyClassifier(strategy="uniform")
model.fit(X, y)
y_pred = model.predict(X)

fbeta_score(y, y_pred, beta=1.5)


# Piewszy wynik jest **31%**. Teraz przygotujmy dane dla `submit`'u!
# 
# W tym celu należy wczytać dane ze zbioru testowego: `../input/diabetic_test.h5`.
# 
# ## 👀 Prognoza `submit`

# In[4]:


test = pd.read_hdf('../input/diabetic_test.h5')
X = test[feats].values

y_pred = model.predict(X)


# In[5]:


train.shape, test.shape


# In[6]:


test['readmitted'] = y_pred
test[ ['id', 'readmitted'] ].to_csv('../output/submit_dummy_model.csv', index=False) 


# ## ☝️ Przygotowanie pliku, który możesz dodać na Kaggle 
# 
# Trzeba przygotować plik, który będzie zawierał: `id` i `readmitted`, następnie zaloguj się na Kaggle, wejdź na stronę konkursu i dodaj swój wynik :). Koniecznie napisz na Slacku, że już "pierwsze koty za płoty" :). 🐱

# In[ ]:


## Dane

W celach optymalizacji szereg column (poniżej) zawierają wartości. Tu możesz zobaczyć słownik, który był użyty `{'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}`
- `metformin`
- `repaglinide`
- `nateglinide`
- `chlorpropamide`
- `glimepiride`
- `acetohexamide`
- `glipizide`
- `glyburide`
- `tolbutamide`
- `pioglitazone`
- `rosiglitazone`
- `acarbose`
- `miglitol`
- `troglitazone`
- `tolazamide`
- `examide`
- `citoglipton`
- `insulin`
- `glyburide-metformin`
- `glipizide-metformin`
- `glimepiride-pioglitazone`
- `metformin-rosiglitazone`
- `metformin-pioglitazone`

Teraz czas na kodowanie...


# ![](../images/submit_dummy.png)

# Teraz już zastosuj swoją wiedzę, którą masz.
# Spróbuj znaleźć lepsze cechy, lepszy model i lepsze parametry.
# 
# Powodzenia!

# In[45]:


import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.metrics import fbeta_score
from sklearn.dummy import DummyClassifier

train = pd.read_hdf('../input/diabetic_train.h5')
print(train.shape)
print(train.info())
print(train.describe())
train.head()


# In[46]:


train = train.copy() 
Rep = train.replace('?', np.NaN) 
nacheck = Rep.isnull().sum() 
nacheck


# In[47]:


train= train.drop(['weight','payer_code','medical_specialty'], axis =1)


# In[48]:


train['readmitted']


# In[53]:


print (train[train['readmitted'] == 0 ].shape)
print (train[train['readmitted'] == 1 ].shape)


# In[54]:


train.groupby('readmitted').size()


# In[55]:


train = train[((train.discharge_disposition_id != 11) & 
                                          (train.discharge_disposition_id != 13) &
                                          (train.discharge_disposition_id != 14) & 
                                          (train.discharge_disposition_id != 19) & 
                                          (train.discharge_disposition_id != 20) & 
                                          (train.discharge_disposition_id != 21))]


# In[56]:


train.head()


# In[65]:


sortage = train.sort_values(by = 'age')
x = sns.stripplot(x = "age", y = "num_medications", data = sortage, color = 'blue')
sns.despine()
x.figure.set_size_inches(10, 6)
x.set_xlabel('Age')
x.set_ylabel('Number of Medications')
x.axes.set_title('Number of Medications vs. Age')
plt.show()


# In[66]:


plot1 = sns.countplot(x = 'gender', hue = 'readmitted', data = train) 

sns.despine()
plot1.figure.set_size_inches(7, 6.5)
plot1.legend(title = 'Readmitted patients', labels = ('No', 'Yes'))
plot1.axes.set_title('Readmissions Balance by Gender')
plt.show()


# In[67]:


b = train.age.unique()
b.sort()
b_sort = np.array(b).tolist()


ageplt = sns.countplot(x = 'age', hue = 'readmitted', data = train, order = b_sort) 

sns.despine()
ageplt.figure.set_size_inches(7, 6.5)
ageplt.legend(title = 'Readmitted within 30 days', labels = ('No', 'Yes'))
ageplt.axes.set_title('Readmissions Balance by Age')
plt.show()


# In[70]:


import seaborn as sns

fig, ax = plt.subplots(figsize=(15,10), ncols=2, nrows=2)

sns.countplot(x="readmitted", data=train, ax=ax[0][0])
sns.countplot(x="race", data=train, ax=ax[0][1])
sns.countplot(x="gender", data=train, ax=ax[1][0])
sns.countplot(x="age", data=train, ax=ax[1][1])


# In[71]:


numcolumn = train.select_dtypes(include = [np.number]).columns
objcolumn = train.select_dtypes(include = ['object']).columns


# In[72]:


train[numcolumn] = train[numcolumn].fillna(0)
train[objcolumn] = train[objcolumn].fillna("unknown")


# In[73]:


train.head(2)


# In[ ]:


def map_now():
    listname = [('infections', 139),
                ('neoplasms', (239 - 139)),
                ('endocrine', (279 - 239)),
                ('blood', (289 - 279)),
                ('mental', (319 - 289)),
                ('nervous', (359 - 319)),
                ('sense', (389 - 359)),
                ('circulatory', (459-389)),
                ('respiratory', (519-459)),
                ('digestive', (579 - 519)),
                ('genitourinary', (629 - 579)),
                ('pregnancy', (679 - 629)),
                ('skin', (709 - 679)),
                ('musculoskeletal', (739 - 709)),
                ('congenital', (759 - 739)),
                ('perinatal', (779 - 759)),
                ('ill-defined', (799 - 779)),
                ('injury', (999 - 799))]
    
    
    dictcout = {}
    count = 1
    for name, num in listname:
        for i in range(num):
            dictcout.update({str(count): name})  
            count += 1
    return dictcout
  

def codemap(df, codes):
    import pandas as pd
    namecol = df.columns.tolist()
    for col in namecol:
        temp = [] 
        for num in df[col]:           
            if ((num is None) | (num in ['unknown', '?']) | (pd.isnull(num))): temp.append('unknown')
            elif(num.upper()[0] == 'V'): temp.append('supplemental')
            elif(num.upper()[0] == 'E'): temp.append('injury')
            else: 
                lkup = num.split('.')[0]
                temp.append(codes[lkup])           
        df.loc[:, col] = temp               
    return df 


listcol = ['diag_1', 'diag_2', 'diag_3']
codes = map_now()
train[listcol] = codemap(train[listcol], codes)


# In[74]:


train.describe


# In[75]:


data1 = train.drop(['encounter_id', "patient_nbr", 'admission_type_id','readmitted'], axis =1)


# In[76]:


data1.head(2)


# In[77]:


listnormal = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                     'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

from sklearn.preprocessing import StandardScaler

normal = StandardScaler()

data1[listnormal] = normal.fit_transform(data1[listnormal])

data1.describe()


# In[85]:


feats = ['encounter_id']
X = train[feats].values
y = train['readmitted'].values

model = DummyClassifier(strategy="uniform")
model.fit(X, y)
y_pred = model.predict(X)

fbeta_score(y, y_pred, beta=1.5)


# In[86]:


test = pd.read_hdf('../input/diabetic_test.h5')
X = test[feats].values

y_pred = model.predict(X)


# In[87]:


train.shape, test.shape


# In[88]:


test['readmitted'] = y_pred
test[ ['id', 'readmitted'] ].to_csv('../output/submit_ctb_treshold0.15.csv', index=False)


# In[ ]:





# In[89]:


import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator
import xgboost as xgb
import catboost as ctb

from tqdm import tqdm


# In[90]:


train = pd.read_hdf('../input/diabetic_train.h5')
train.info()


# In[91]:


def __f15_score(y, y_pred):
    return fbeta_score(y, y_pred, beta=1.5)


f15_score  = make_scorer(__f15_score, greater_is_better=True)


# In[92]:


def get_feats(df, black_list=["readmitted", "id"]):
    feats = df.select_dtypes("number").columns
    feats = [x for x in feats if x not in black_list]
    
    return feats


def get_X_y(df, feats=None):
    if feats is None:
        feats = get_feats(df)

    X = df[feats].values
    y = df['readmitted'].values
    
    return X, y


# In[93]:


feats = get_feats(train)
len(feats)


# In[100]:


X, y = get_X_y(train)

models = [
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=7, n_estimators=100, random_state=0),
    xgb.XGBClassifier(max_depth=7, n_estimators=100, learning_rate=0.3, random_state=0),
    ctb.CatBoostClassifier(max_depth=7, n_estimators=100, learning_rate=0.3, random_state=0, verbose=False)
    
]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

for model in models:
    model_str = str(model.__class__).split(".")[-1].split("'")[0]
    scores = cross_val_score(model, X, y, cv=cv, scoring=f15_score)
    print(f"{model_str} f1.5={np.mean(scores):.2f} std={np.std(scores):.2f}")


# In[95]:


model = ctb.CatBoostClassifier(max_depth=7, n_estimators=100, learning_rate=0.3, random_state=0, verbose=False)

X, y = get_X_y(train)
scores = []
for train_idx, test_idx in tqdm(cv.split(X, y, groups=train["patient_nbr"])):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    score =  __f15_score(y_test, y_pred)
    scores.append(score)
    
np.mean(scores)


# In[96]:


model = ctb.CatBoostClassifier(max_depth=7, n_estimators=100, learning_rate=0.3, random_state=0, verbose=False)

threshold = 0.15
X, y = get_X_y(train)
scores = []
for train_idx, test_idx in tqdm(cv.split(X, y, groups=train["patient_nbr"])):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype("int8")
    
    score =  __f15_score(y_test, y_pred)
    scores.append(score)
    
np.mean(scores)


# In[97]:


test = pd.read_hdf('../input/diabetic_test.h5')


# In[98]:


threshold = 0.15
feats = get_feats(train)

X_train, y_train = get_X_y(train, feats)
X_test = test[feats].values

model = ctb.CatBoostClassifier(max_depth=7, n_estimators=100, learning_rate=0.3, random_state=0, verbose=False)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > threshold).astype("int8")


# In[102]:


test['readmitted'] = y_pred
test[ ['id', 'readmitted'] ].to_csv('../output/submit_ctb_treshold0.15.csv', index=False)


# ## 🧠 Przydatne linki
# 1. [A Look at Precision, Recall, and F1-Score](https://towardsdatascience.com/a-look-at-precision-recall-and-f1-score-36b5fd0dd3ec)
# 2. [Beyond the F-1 score: A look at the F-beta score](https://medium.com/@douglaspsteen/beyond-the-f-1-score-a-look-at-the-f-beta-score-3743ac2ef6e3)

# ## ⭐ Dlaczego warto wykonać to zadanie?
# 
# Potraktuj to zadanie jako budowanie rozwiązania, które może trafić do Twojego portfolio :) 
# To bardzo ważne, aby umieć przekuć wiedzę zdobytą w kursie umieć przekuć na rozwiązanie konkretnego problemu z pomocą uczenia maszynowego. To nie musi być doskonałe. Ważne, aby to zrobić i umieć to pokazać. 
# 
# Większość absolwentów naszych kursów stwierdziła, że to właśnie konkurs podczas kursu, czy samodzielna praca pozwoliła im uporządkować wiedzę, którą zdobyli i w ciągu kilku dni niesamowicie posunać się do przodu. 
# 
# Początki będą ciężkie i mogą być demotywujące, ale regularność działania pomoże Ci to przejść, a efekatmi będziesz zaskoczona/y. 
