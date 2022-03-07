# predict_immigration
Predictive analysis for immigration status in the US using CPS data



```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


```


```python
df = pd.read_csv('CPSdata.csv')
```

    /Users/imgesucetin/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3020: DtypeWarning: Columns (5) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
df['citizen'].unique()
```




    array(['Naturalized citizen', 'Born in U.S', 'Not a citizen',
           'Born abroad of American parents', 'Born in U.S. outlying'],
          dtype=object)




```python
df['immigrant'] = 0
```


```python
df.loc[df['citizen']=='Naturalized citizen', 'immigrant'] = 1
df.loc[df['citizen']=='Not a citizen', 'immigrant'] = 1
```


```python
df.loc[df['age']=='Under 1 year ', 'age'] = 1
df['age'] = pd.to_numeric(df.age, errors='coerce').fillna(0).astype(np.int64)
```


```python
df.columns
```




    Index(['year', 'serial', 'month', 'cpsid', 'asecflag', 'hflag', 'asecwth',
           'statefip', 'metarea', 'pernum', 'cpsidp', 'asecwt', 'age', 'sex',
           'bpl', 'yrimmig', 'citizen', 'mbpl', 'fbpl', 'nativity', 'empstat',
           'labforce', 'occ', 'ind', 'educ', 'ftotval', 'inctot', 'incwage',
           'immigrant'],
          dtype='object')




```python
#df[['citizen','immigrant']]
```


```python
df.drop(['asecflag','serial','hflag','cpsidp','bpl','yrimmig','citizen','nativity', 'inctot', 'incwage'], axis=1, inplace=True)
```


```python
df.columns
```




    Index(['year', 'month', 'cpsid', 'asecwth', 'statefip', 'metarea', 'pernum',
           'asecwt', 'age', 'sex', 'mbpl', 'fbpl', 'empstat', 'labforce', 'occ',
           'ind', 'educ', 'ftotval', 'immigrant'],
          dtype='object')




```python
#df['metarea'].unique()
```


```python
df['statefip22'] = pd.Categorical(df.statefip)
df['metarea_cat'] = pd.Categorical(df.metarea)
df['educ_cat'] = pd.Categorical(df.educ)
df['labforce_cat'] = pd.Categorical(df.labforce)
df['empstat_cat'] = pd.Categorical(df.empstat)
df['sex_cat'] = pd.Categorical(df.sex)
df['mbpl_cat'] = pd.Categorical(df.mbpl)
df['fbpl_cat'] = pd.Categorical(df.fbpl)
df['occ'] = pd.Categorical(df.occ)
df['ind'] = pd.Categorical(df.ind)

```

### Apply LabelEncoder to categorical features


```python
from sklearn.preprocessing import LabelEncoder
cols = ('statefip22','metarea_cat','educ_cat','labforce_cat','empstat_cat',
        'sex_cat','mbpl_cat','fbpl_cat')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df[c].values)) 
    df[c] = lbl.transform(list(df[c].values))
```


```python
print('Shape df: {}'.format(df.shape))
```

    Shape df: (4454691, 27)



```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>month</th>
      <th>cpsid</th>
      <th>asecwth</th>
      <th>statefip</th>
      <th>metarea</th>
      <th>pernum</th>
      <th>asecwt</th>
      <th>age</th>
      <th>sex</th>
      <th>...</th>
      <th>ftotval</th>
      <th>immigrant</th>
      <th>statefip22</th>
      <th>metarea_cat</th>
      <th>educ_cat</th>
      <th>labforce_cat</th>
      <th>empstat_cat</th>
      <th>sex_cat</th>
      <th>mbpl_cat</th>
      <th>fbpl_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415800</td>
      <td>878.58</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>878.58</td>
      <td>76</td>
      <td>Male</td>
      <td>...</td>
      <td>4524</td>
      <td>1</td>
      <td>19</td>
      <td>339</td>
      <td>10</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>59</td>
      <td>61</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415800</td>
      <td>878.58</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>878.58</td>
      <td>70</td>
      <td>Female</td>
      <td>...</td>
      <td>4524</td>
      <td>1</td>
      <td>19</td>
      <td>339</td>
      <td>8</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>59</td>
      <td>61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415700</td>
      <td>841.89</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>841.89</td>
      <td>41</td>
      <td>Male</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>11</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>155</td>
      <td>78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415600</td>
      <td>878.58</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>878.58</td>
      <td>71</td>
      <td>Male</td>
      <td>...</td>
      <td>29800</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>16</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415600</td>
      <td>878.58</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>878.58</td>
      <td>71</td>
      <td>Female</td>
      <td>...</td>
      <td>29800</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415500</td>
      <td>838.36</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>838.36</td>
      <td>40</td>
      <td>Female</td>
      <td>...</td>
      <td>63350</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415500</td>
      <td>838.36</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>838.36</td>
      <td>44</td>
      <td>Male</td>
      <td>...</td>
      <td>63350</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415500</td>
      <td>838.36</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>3</td>
      <td>1043.34</td>
      <td>21</td>
      <td>Male</td>
      <td>...</td>
      <td>63350</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>16</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415500</td>
      <td>838.36</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>4</td>
      <td>884.61</td>
      <td>17</td>
      <td>Male</td>
      <td>...</td>
      <td>63350</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415500</td>
      <td>838.36</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>5</td>
      <td>827.13</td>
      <td>2</td>
      <td>Female</td>
      <td>...</td>
      <td>63350</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415400</td>
      <td>566.21</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>566.21</td>
      <td>51</td>
      <td>Male</td>
      <td>...</td>
      <td>40830</td>
      <td>1</td>
      <td>19</td>
      <td>339</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>27</td>
      <td>28</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415400</td>
      <td>566.21</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>566.21</td>
      <td>41</td>
      <td>Female</td>
      <td>...</td>
      <td>40830</td>
      <td>1</td>
      <td>19</td>
      <td>339</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>27</td>
      <td>28</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415400</td>
      <td>566.21</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>3</td>
      <td>566.21</td>
      <td>16</td>
      <td>Female</td>
      <td>...</td>
      <td>40830</td>
      <td>1</td>
      <td>19</td>
      <td>339</td>
      <td>5</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>27</td>
      <td>28</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415400</td>
      <td>566.21</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>4</td>
      <td>578.82</td>
      <td>13</td>
      <td>Female</td>
      <td>...</td>
      <td>40830</td>
      <td>1</td>
      <td>19</td>
      <td>339</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>27</td>
      <td>28</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415400</td>
      <td>566.21</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>5</td>
      <td>578.82</td>
      <td>9</td>
      <td>Female</td>
      <td>...</td>
      <td>40830</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>27</td>
      <td>28</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415300</td>
      <td>959.87</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>959.87</td>
      <td>28</td>
      <td>Male</td>
      <td>...</td>
      <td>41400</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415300</td>
      <td>959.87</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>893.33</td>
      <td>26</td>
      <td>Male</td>
      <td>...</td>
      <td>40400</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415200</td>
      <td>838.36</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>838.36</td>
      <td>44</td>
      <td>Female</td>
      <td>...</td>
      <td>27440</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>15</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415100</td>
      <td>866.76</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>866.76</td>
      <td>81</td>
      <td>Male</td>
      <td>...</td>
      <td>71666</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>15</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>112</td>
      <td>157</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415100</td>
      <td>866.76</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>866.76</td>
      <td>77</td>
      <td>Female</td>
      <td>...</td>
      <td>71666</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>12</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415000</td>
      <td>902.12</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>902.12</td>
      <td>45</td>
      <td>Male</td>
      <td>...</td>
      <td>35781</td>
      <td>1</td>
      <td>19</td>
      <td>339</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>48</td>
      <td>50</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415000</td>
      <td>902.12</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>902.12</td>
      <td>36</td>
      <td>Female</td>
      <td>...</td>
      <td>35781</td>
      <td>1</td>
      <td>19</td>
      <td>339</td>
      <td>16</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>48</td>
      <td>50</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415000</td>
      <td>902.12</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>3</td>
      <td>890.91</td>
      <td>11</td>
      <td>Male</td>
      <td>...</td>
      <td>35781</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>48</td>
      <td>50</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415000</td>
      <td>902.12</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>4</td>
      <td>1022.79</td>
      <td>9</td>
      <td>Male</td>
      <td>...</td>
      <td>35781</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>48</td>
      <td>50</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302415000</td>
      <td>902.12</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>5</td>
      <td>842.41</td>
      <td>8</td>
      <td>Male</td>
      <td>...</td>
      <td>35781</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>48</td>
      <td>50</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302414900</td>
      <td>879.58</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>879.58</td>
      <td>37</td>
      <td>Male</td>
      <td>...</td>
      <td>31050</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302414800</td>
      <td>832.52</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>832.52</td>
      <td>32</td>
      <td>Male</td>
      <td>...</td>
      <td>5898</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>10</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302414800</td>
      <td>832.52</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>953.64</td>
      <td>24</td>
      <td>Female</td>
      <td>...</td>
      <td>3520</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>11</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302414700</td>
      <td>816.85</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>1</td>
      <td>816.85</td>
      <td>69</td>
      <td>Male</td>
      <td>...</td>
      <td>24539</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>11</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1994</td>
      <td>March</td>
      <td>19940302414700</td>
      <td>816.85</td>
      <td>Maine</td>
      <td>Portland, ME</td>
      <td>2</td>
      <td>816.85</td>
      <td>62</td>
      <td>Female</td>
      <td>...</td>
      <td>24539</td>
      <td>0</td>
      <td>19</td>
      <td>339</td>
      <td>8</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>155</td>
      <td>30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4454661</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>542.00</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>2</td>
      <td>542.00</td>
      <td>27</td>
      <td>Female</td>
      <td>...</td>
      <td>35004</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454662</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>382.46</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>382.46</td>
      <td>57</td>
      <td>Male</td>
      <td>...</td>
      <td>30010</td>
      <td>1</td>
      <td>11</td>
      <td>433</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>67</td>
      <td>69</td>
    </tr>
    <tr>
      <th>4454663</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>603.62</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>603.62</td>
      <td>35</td>
      <td>Female</td>
      <td>...</td>
      <td>87018</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>120</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454664</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>426.10</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>426.10</td>
      <td>31</td>
      <td>Male</td>
      <td>...</td>
      <td>102458</td>
      <td>1</td>
      <td>11</td>
      <td>433</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>79</td>
      <td>81</td>
    </tr>
    <tr>
      <th>4454665</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>344.19</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>344.19</td>
      <td>66</td>
      <td>Male</td>
      <td>...</td>
      <td>73639</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>11</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454666</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>344.19</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>2</td>
      <td>344.19</td>
      <td>56</td>
      <td>Female</td>
      <td>...</td>
      <td>73639</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>16</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454667</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>448.12</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>448.12</td>
      <td>35</td>
      <td>Male</td>
      <td>...</td>
      <td>38000</td>
      <td>1</td>
      <td>11</td>
      <td>433</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>65</td>
      <td>67</td>
    </tr>
    <tr>
      <th>4454668</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>448.12</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>2</td>
      <td>448.12</td>
      <td>36</td>
      <td>Female</td>
      <td>...</td>
      <td>38000</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>16</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454669</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>448.12</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>3</td>
      <td>533.38</td>
      <td>14</td>
      <td>Female</td>
      <td>...</td>
      <td>38000</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454670</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>448.12</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>4</td>
      <td>418.36</td>
      <td>13</td>
      <td>Male</td>
      <td>...</td>
      <td>38000</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454671</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>448.12</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>5</td>
      <td>528.06</td>
      <td>10</td>
      <td>Female</td>
      <td>...</td>
      <td>38000</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454672</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>448.12</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>6</td>
      <td>461.95</td>
      <td>5</td>
      <td>Male</td>
      <td>...</td>
      <td>38000</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454673</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>376.55</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>376.55</td>
      <td>41</td>
      <td>Female</td>
      <td>...</td>
      <td>76300</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>16</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454674</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>376.55</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>2</td>
      <td>376.55</td>
      <td>38</td>
      <td>Male</td>
      <td>...</td>
      <td>76300</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>95</td>
      <td>97</td>
    </tr>
    <tr>
      <th>4454675</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>376.55</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>3</td>
      <td>369.99</td>
      <td>13</td>
      <td>Male</td>
      <td>...</td>
      <td>76300</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454676</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>376.55</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>4</td>
      <td>269.67</td>
      <td>10</td>
      <td>Female</td>
      <td>...</td>
      <td>76300</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454677</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>400.70</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>400.70</td>
      <td>85</td>
      <td>Female</td>
      <td>...</td>
      <td>55336</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>12</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>155</td>
      <td>85</td>
    </tr>
    <tr>
      <th>4454678</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>325.01</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>325.01</td>
      <td>85</td>
      <td>Male</td>
      <td>...</td>
      <td>13800</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>11</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454679</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>357.47</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>357.47</td>
      <td>80</td>
      <td>Female</td>
      <td>...</td>
      <td>26272</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>11</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454680</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>372.07</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>372.07</td>
      <td>58</td>
      <td>Male</td>
      <td>...</td>
      <td>173455</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>12</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454681</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>372.07</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>2</td>
      <td>372.07</td>
      <td>54</td>
      <td>Female</td>
      <td>...</td>
      <td>173455</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454682</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>372.07</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>3</td>
      <td>443.48</td>
      <td>20</td>
      <td>Male</td>
      <td>...</td>
      <td>173455</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>16</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454683</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>372.07</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>4</td>
      <td>325.03</td>
      <td>80</td>
      <td>Female</td>
      <td>...</td>
      <td>173455</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>11</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>79</td>
      <td>81</td>
    </tr>
    <tr>
      <th>4454684</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>310.56</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>310.56</td>
      <td>62</td>
      <td>Male</td>
      <td>...</td>
      <td>146902</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454685</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>310.56</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>2</td>
      <td>310.56</td>
      <td>50</td>
      <td>Female</td>
      <td>...</td>
      <td>126</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454686</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>521.16</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>521.16</td>
      <td>25</td>
      <td>Female</td>
      <td>...</td>
      <td>12440</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454687</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>309.74</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>309.74</td>
      <td>71</td>
      <td>Female</td>
      <td>...</td>
      <td>17401</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>155</td>
      <td>122</td>
    </tr>
    <tr>
      <th>4454688</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>355.09</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>355.09</td>
      <td>61</td>
      <td>Female</td>
      <td>...</td>
      <td>66836</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>12</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>157</td>
    </tr>
    <tr>
      <th>4454689</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>355.09</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>2</td>
      <td>414.32</td>
      <td>19</td>
      <td>Female</td>
      <td>...</td>
      <td>66836</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>16</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>35</td>
    </tr>
    <tr>
      <th>4454690</th>
      <td>2017</td>
      <td>March</td>
      <td>0</td>
      <td>333.11</td>
      <td>Hawaii</td>
      <td>Urban Honolulu, HI</td>
      <td>1</td>
      <td>333.11</td>
      <td>44</td>
      <td>Female</td>
      <td>...</td>
      <td>82000</td>
      <td>0</td>
      <td>11</td>
      <td>433</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>148</td>
      <td>157</td>
    </tr>
  </tbody>
</table>
<p>4454691 rows × 27 columns</p>
</div>




```python
print(df['ftotval'].max())
print(df['ftotval'].min())
```

    2742997
    -37040



```python
df_size = len(df['year'])
```


```python
test_start_date = 2013
feats = ('statefip_cat','metarea_cat','educ_cat','labforce_cat',
         'empstat_cat','sex_cat','mbpl_cat', 'fbpl_cat', 
         'asecwth','pernum','asecwt', 'age','occ','ind','ftotval')


train_ind = (df['year'] < test_start_date)
X_train=df.loc[train_ind, feats]
y_train=df.loc[train_ind, 'immigrant']
train_size = len(X_train)

test_ind = (df['year'] >= test_start_date)
X_test=df.loc[test_ind,feats]
y_test=df.loc[test_ind,'immigrant']
test_size = len(y_test)

print(train_size/df_size)
print(test_size/df_size)
```

    /Users/imgesucetin/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:870: FutureWarning: 
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.
    
    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
      return self._getitem_lowerdim(tup)


    0.7816649909050931
    0.2183350090949069



```python
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)
```

    (3482076, 15)
    (3482076,)
    (972615, 15)
    (972615,)



```python
from sklearn.ensemble import RandomForestClassifier

# train model
model = RandomForestClassifier(n_estimators= 25, max_depth= None,max_features = 0.4,random_state= 11 )
#fit the data
model.fit(X_train, y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-20-92b31c4023e7> in <module>
          4 model = RandomForestClassifier(n_estimators= 25, max_depth= None,max_features = 0.4,random_state= 11 )
          5 #fit the data
    ----> 6 model.fit(X_train, y_train)
    

    ~/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py in fit(self, X, y, sample_weight)
        248 
        249         # Validate or convert input data
    --> 250         X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        251         y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        252         if sample_weight is not None:


    ~/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        571         if force_all_finite:
        572             _assert_all_finite(array,
    --> 573                                allow_nan=force_all_finite == 'allow-nan')
        574 
        575     shape_repr = _shape_repr(array.shape)


    ~/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan)
         54                 not allow_nan and not np.isfinite(X).all()):
         55             type_err = 'infinity' if allow_nan else 'NaN, infinity'
    ---> 56             raise ValueError(msg_err.format(type_err, X.dtype))
         57 
         58 


    ValueError: Input contains NaN, infinity or a value too large for dtype('float32').



```python
df['predict'] = -1

df.loc[train_ind, 'predict'] = model.predict(X_train)
df.loc[test_ind, 'predict'] = model.predict(X_test)
```


```python
df
```


```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

conf = confusion_matrix(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'])
test_acc = accuracy_score(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'])
precision, recall, fscore, support = score(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'])
print('accuracy: {}'.format(test_acc))
print('recall: {}'.format(recall)) # true positive rate 
print('precision: {}'.format(precision))
```


```python

```


```python
conf



```


```python
df.to_csv('CPS_Results.csv')
```


```python
import matplotlib

import matplotlib.pyplot as plt
%matplotlib inline

plt.xlabel('Feature Importance on Immigration Status')
plt.ylabel('Features of My Model')

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')

plt.savefig('independentvariables.png')

```


```python
np.set_printoptions(precision=2)

class_names= np.array(['NonImmigrant','Immigrant'])
print(type(class_names))

# Plot non-normalized confusion matrix
plot_confusion_matrix(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'], classes=class_names,
                      title='Confusion matrix, Counts')

# Plot normalized confusion matrix
plot_confusion_matrix(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'], classes=class_names, normalize=True,
                      title='Confusion matrix, Percentage')

plt.show()
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
test_start_date = 2013
feats = ('statefip_cat','metarea_cat','educ_cat','labforce_cat',
         'empstat_cat','sex_cat', 'asecwth','pernum','asecwt', 'age','occ','ind','ftotval')


train_ind = (df['year'] < test_start_date)
X_train=df.loc[train_ind, feats]
y_train=df.loc[train_ind, 'immigrant']
train_size = len(X_train)

test_ind = (df['year'] >= test_start_date)
X_test=df.loc[test_ind,feats]
y_test=df.loc[test_ind,'immigrant']
test_size = len(y_test)

print(train_size/df_size)
print(test_size/df_size)
```


```python

```


```python
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)
```


```python
from sklearn.ensemble import RandomForestClassifier

# train model
model = RandomForestClassifier(n_estimators= 25, max_depth= None,max_features = 0.4,random_state= 11 )
#fit the data
model.fit(X_train, y_train)
```


```python
df['predict'] = -1

df.loc[train_ind, 'predict'] = model.predict(X_train)
df.loc[test_ind, 'predict'] = model.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

conf = confusion_matrix(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'])
test_acc = accuracy_score(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'])
precision, recall, fscore, support = score(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'])
print('accuracy: {}'.format(test_acc))
print('recall: {}'.format(recall)) # true positive rate 
print('precision: {}'.format(precision))
```


```python
conf
```


```python
df.to_csv('CPS_Results.csv')
```


```python
%matplotlib inline

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')



```


```python

import matplotlib.pyplot as plt
import numpy as np
label = ['Adventure', 'Action', 'Drama', 'Comedy', 'Thriller/Suspense', 'Horror', 'Romantic Comedy', 'Musical',
         'Documentary', 'Black Comedy', 'Western', 'Concert/Performance', 'Multiple Genres', 'Reality']

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, no_movies)
    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('No of Movies', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Market Share for Each Genre 1995-2017')
    plt.show()
```


```python
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('train_ind')
y_pos = np.arange(len(objects))
performance = []
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('')
plt.title('')
 
plt.show()
```


```python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

income = 
expenditure = 2

X = np.array(income)
Y = np.array(expenditure)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
model = LinearRegression()
model.fit(X_train, Y_train)
print("The intercept of the model: {}".format(model.intercept_))
print("Coefficient of variable: {}".format(model.coef_))

# The sum squared error
print("Sum of squared error: {}".format(np.sum(model.predict(X_test) - Y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: {}'.format(model.score(X_test, Y_test)))
```


```python
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
```


```python
np.set_printoptions(precision=2)

class_names= np.array(['NonImmigrant','Immigrant'])
print(type(class_names))

# Plot non-normalized confusion matrix
plot_confusion_matrix(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'], classes=class_names,
                      title='Confusion matrix, Counts')

# Plot normalized confusion matrix
plot_confusion_matrix(df.loc[test_ind,'immigrant'], df.loc[test_ind,'predict'], classes=class_names, normalize=True,
                      title='Confusion matrix, Percentage')

plt.show()

```


```python
df.loc['immigrant'==0]
```


```python
td=df.loc[test_ind,'immigrant']
print(len(td))
3
print(len(td.loc[td==1]))
```


```python

```
