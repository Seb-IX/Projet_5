import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dtypes(dataframe,figsize=(12,10)):
    """Affiche un graphique de la répartition des type de variables d'un pd.DataFrame.
        Parameters
        ----------
        dataframe : pd.DataFrame,
            Pandas DataFrame.
            
        Optional
        ----------    
        figsize : tulpe, default=(12,10)
            Indique les dimensions du graphique
    """
    print(dataframe.dtypes.value_counts())
    plt.figure(figsize=figsize)
    plt.pie(dataframe.dtypes.value_counts().values,autopct="%1.2f%%",labels=[str(types) for types in dataframe.dtypes.value_counts().index])
    plt.title("Répartition des types dans le jeu de données.")
    plt.ylabel("Type des données")
    plt.legend()
    plt.show()

def analyse_num(dataframe,remove=["Col_ID"],hist_bins=20):
    """Fait une analyse global du dataframe en affichant la répartition des variables continue à l'aide d'un histogramme et un boxplot.
        Parameters
        ----------
        dataframe : pd.DataFrame,
            Pandas DataFrame.
            
        Optional
        ----------    
        hist_bins : int, default=20
            Le nombre de segment de l'histogramme
            
        remove : list, default=["Col_ID"]
            Permet de retirer des colonnes qui n'ont aucune utilité ou qui ne sont pas des type float ou entier.
    """
    num_cols = [col for col in dataframe.columns if not col in dataframe.select_dtypes(["object","category"]).columns.to_list()]
    if len(remove) > 0:
        num_cols = [col for col in num_cols if (col not in remove)]

    for col in num_cols:
        fig,axes = plt.subplots(1,2,figsize=(20,5))
        dataframe.hist(str(col),ax=axes[0],bins=hist_bins)
        dataframe.boxplot(str(col),ax=axes[1], vert=False)
        
        axes[1].set_yticklabels([])
        axes[1].set_yticks([])
        axes[0].set_title(col + " | Histogram")
        axes[1].set_title(col + " | Boxplot")
        plt.show()
        
def missing_value(dataframe,reversed_sort=False,head=None,printing=False,heatmap=False):
    """Affiche la répartition des lignes vide du jeu de données, avec la possiblité d'afficher un graphique en carte de chaleur des lignes vides et un tableau de répartition des lignes vides.
        Parameters
        ----------
        dataframe : pd.DataFrame,
            Pandas DataFrame.
            
        Optional
        ----------    
        reversed_sort : bool, default=False
            Permet de triée le tableaux dans l'ordre des colonne les plus vide au moins vide et inversement si égale à True.
        
        head : int, default=None
            Permet de renvoyer le tableaux avec uniquement les 'head' premières lignes si égale à None, alors la fonction renvoie tout le tableaux.
        
        printing : bool, default=False
            Permet de print le tableau de valeur vide
            
        heatmap : bool, default=False
            Affiche la carte de chaleur des lignes vide du jeu de données
            
        Returns
        -------
        df, pd.DataFrame,
            DataFrame avec 'head' lignes.
         ______________________________________________________________
        | Colonne | Nombre de lignes vide | Pourcentage de lignes vide |
        |_________|_______________________|____________________________|
        | col_n   |         1000          |             10             |
        |_________|_______________________|____________________________|
    """
    if type(reversed_sort) != bool:
        reversed_sort=False
    miss_value = dataframe.isna().sum().sort_values(ascending =reversed_sort)
    miss_value = pd.DataFrame(miss_value[miss_value.values > 0],columns=["Nombre de valeurs vides"])
    miss_value["Pourcentage sur le total"] = round((miss_value["Nombre de valeurs vides"] / dataframe.shape[0])*100,4)
    
    if heatmap:
        plt.figure(figsize=(20,10))
        sns.heatmap(dataframe.isna(),cbar=False,)
        plt.title("répartition des valeurs vides")
        plt.show()
    
    if miss_value.shape[0] == 0:
        print("Aucune valeur manquante.")
        return None
    
    if head != None and type(head) == int :
        print(f"Il y a {miss_value.shape[0]} colonnes avec des valeurs manquantes qui sont (pour les {head} premiers):")
        if printing:
            print(miss_value.head(head))
        return miss_value.head(head)
    else:
        print(f"Il y a {miss_value.shape[0]} colonnes avec des valeurs manquantes qui sont :")
        if printing :
            print(miss_value)
        return miss_value
    
def presentation_df(df,head=5):
    """Affiche les dimensionnalité du DataFrame, et les 'head' premières lignes du DataFrame.
        Parameters
        ----------
        df : pd.DataFrame,
            Pandas DataFrame.
            
        Optional
        ----------    
        head : int, default=5
            Indique les dimensions du graphique
            
        Returns
        -------
        df, pd.DataFrame,
            DataFrame avec 'head' lignes.
    """
    print("Dimensionnalitées :",df.shape)
    return df.head(head)

def describe_cat_df(df):
    """Permet de faire une analyse descriptive des variables catégoriel en affichant, le nombres de valeurs non vide, les valeur unique, la valeur la plus courante, la fréquence de cette valeur, et le nombre de valeurs uniques. 
        Parameters
        ----------
        df : pd.DataFrame,
            Pandas DataFrame.
            
        Returns
        -------
        cat_df, pd.DataFrame,
            Présentation des différents colonnes catégorielles sous la forme :
         __________________________________________________________________________
        | Colonne | count | unique              | top          | freq   | nunique  |
        |_________|_______|_____________________|______________|________|__________|
        | GENDER  | 20000 | [M,F]               | M            | 11000  | 2        |
        |_________|_______|_____________________|______________|________|__________|
    """
    cat_df = df.select_dtypes("object").describe().T
    cat_df["unique"] = df.select_dtypes("object").apply( pd.Series.unique, axis=0)
    cat_df["nunique"] = df.select_dtypes("object").apply( pd.Series.nunique, axis=0)
    return cat_df

