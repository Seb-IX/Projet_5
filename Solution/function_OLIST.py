from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import  PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

from yellowbrick.cluster import SilhouetteVisualizer

# style par défaut
style_plot = "seaborn-darkgrid"
plt.style.use(style_plot)

###### EDA

def print_interval_qcut(df_cut_col,match_tab_quartile = None, init_msg=None):
    """Retourne l'interval effectuée avec la fonction pandas 'qcut', sous forme => 0 : [0, 10[.
        Parameters
        ----------
        df_cut_col : pd.Series,
            Series pandas contenant la colonne qui à était segmenté par la fonction 'qcut' .
            
        Optional
        ----------    
        match_tab_quartile : array(int), default=None
            Permet de spécifier la liste qui à était attribué par 'qcut' 
            pour mettre la correspondance des intervalles.
        
        init_msg : str, default=None
            Permet d'afficher un message juste avant la mise en forme de l'interval.
            
        Returns
        -------
        msg, str,
            Message des différentes intervalles effectuée par 'qcut' sous la forme suivante :
            '1: (0, 1] 
             2: (1, 2]
             3: (2, inf.)'.
    """
    if not init_msg is None:
        msg = init_msg
    else:
        msg = ""
        
    size=len(df_cut_col.unique())
    
    for i,c in zip(range(size),
                   df_cut_col.sort_values().astype(str).unique()):
        
        if not match_tab_quartile is None:
            msg += "\n" + str(match_tab_quartile[i]) + ": " + c
        else:
            msg+=  "\n" + c
            
    return msg

def describe_cat(df,col,
                 plot=False,printing=True,
                 rotation_xtick=90,
                 rotation_label=0,
                 figsize=(9,7),
                 rounded=3,
                 plot_title="Répartition de la variable",
                 log_scale=False):
    """Permet de détail la répartition d'une colonne catégoriel ou d'entier avec peu de valeur unique.
        Parameters
        ----------
        df : pd.DataFrame,
            pandas DataFrame du jeu de données ou la colonne catégoriel est présente
            
        col : str, 
            Nom de la colonne catégoriel
            
        Optional
        ----------
        plot : bool, default=False
            Indique si la function doit afficher le graphique de la répartition de la variable.
            
        printing : bool, default=True
            Indique si la function doit afficher le tableau de la répartition de la variable.
        
        rotation_xtick : int, default=90
            Indique la rotation en pourcentage(°) du taux afficher sur le graphique
            
        rotation_label : int, default=0
            Indique la rotation en pourcentage(°) du label afficher sur le graphique
            
        figsize : tulpe, default=(9,7)
            Indique les dimensions du graphique.
            
        rounded : int, default=3
            Arrondi les chiffres de répartition.
            
        plot_title : str, default='Répartition de la variable'
            Permet d'ajouter un titre au graphique
           
        log_scale : bool, default=False
            Permet de mettre l'échelle des fréquence sur le graphique au logarithme
            
        Returns
        -------
        error_message, str
           Uniquement si le nombre de valeur unique est supérieur à un seuil
    """
    limit = 25
    
    if df[col].value_counts().shape[0] > limit:
        return "To many category ! (over " + str(limit) +")" 
    
    print(col)
    if printing :
        print(pd.DataFrame({"COUNT":df[col].value_counts(),
                            "RATE":round(df[col].value_counts(normalize=True) *100,rounded).astype(str) + "%"}))
    if plot:
        plt.style.use(style_plot)
        fig,ax = plt.subplots(figsize=figsize)

        bar_x = [i + 1 for i in range(len(df[col].value_counts()))]
        bar_height = [p for p in df[col].value_counts().values]
        bar_tick_label = [label for label in df[col].value_counts().index]
        bar_label = [str(round(p,rounded))+"%" for p in df[col].value_counts(normalize=True).values * 100]
       
        if plot_title != "":
            plt.title(plot_title)
        
        bar_plot = plt.bar(bar_x,bar_height,tick_label=bar_tick_label)
        
        for idx,rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    bar_label[idx],
                    ha='center', va='bottom', rotation=rotation_label)
        
        if log_scale:
            plt.yscale("log")
            
        plt.xticks(rotation=90)

        plt.show()
        
def plot_bar(df,col, rounded=3,
             rotation_label=0,
             rotation_ticks=90,
             plot_title=None,
             figsize=(20,10)):
    """Permet d'afficher un graphique à bar d'une colonne d'un jeu de donnée.
        Parameters
        ----------
        df : pd.DataFrame,
            pandas DataFrame du jeu de données.
            
        col : str, 
            Nom de la colonne à afficher.
            
        Optional
        ----------        
        rotation_xtick : int, default=90
            Indique la rotation en pourcentage(°) du taux afficher sur le graphique
            
        rotation_label : int, default=0
            Indique la rotation en pourcentage(°) du label afficher sur le graphique
            
        figsize : tulpe, default=(9,7)
            Indique les dimensions du graphique.
            
        rounded : int, default=3
            Arrondi les chiffres de répartition.
            
        plot_title : str, default=None
            Permet d'ajouter un titre au graphique
    """
    plt.style.use(style_plot)
    fig,ax = plt.subplots(figsize=figsize)

    bar_x = [i + 1 for i in range(len(df[col].value_counts()))]
    bar_height = [p for p in df[col].values]
    bar_tick_label = [label for label in df[col].index]
    bar_label = [str(round(p,rounded)) for p in df[col].values]

    bar_plot = plt.bar(bar_x,bar_height,tick_label=bar_tick_label)
    
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=rotation_label,fontsize=13)
        
    if not plot_title is None:
        plt.title(plot_title,fontdict={
            "fontsize":15
        })
    plt.xticks(rotation=rotation_ticks,fontsize=13)

    plt.show()

###### plot clustering analyse 

def plot_score_cluster(X,max_cluster=10,max_iter=300,figsize=(12,10)):
    """Affiche un graphique de scoring de l'inertie et du silhouette_score pour un algorithme KMeans (sklearn) initialisé par défaut (init = kmeans++).
        Parameters
        ----------
        X : pd.DataFrame / np.array,
            Jeu de données d'apprentissage. 
            
        Optional
        ----------        
        max_cluster : int, default=10
            Nombre max de cluster de 2 à max_cluster.
            
        max_iter : int, default=300
            Paramètre KMeans, permet de réduire le temps de calcul en limitant le nombre d'itération.
            
        figsize : tulpe, default=(9,7)
            Indique les dimensions du graphique.
    """
    score_silhouette=[]
    score_inertie=[]
    for k in range(2,max_cluster):
        kmean = KMeans(n_clusters=k,max_iter=max_iter)
        kmean.fit(X)
        score_silhouette.append(silhouette_score(X,kmean.labels_))
        score_inertie.append(kmean.inertia_)
    
    plt.style.use(style_plot)
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xlabel("K_cluster")
    ax1.set_ylabel("Silhouette-score",color='tab:red')
    ax1.plot([i for i in range(2,10)], score_silhouette, color='tab:red',marker="o")
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  # instancie un nouvelle axe qui partage l'axe X

    ax2.set_ylabel('Inertia', color='tab:blue')  
    ax2.plot([i for i in range(2,10)], score_inertie, color='tab:blue' ,marker="o")
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Distortion et silhoutte score")

    fig.tight_layout() 
    plt.show()

def plot_snake_comparatif(X,
                          dataframe,
                          segment_col,
                          cluster_col,
                          col_name_X=None,
                          figsize=(20,8)):
    """Permet d'afficher un graphique 'Snake' du jeu de donnée afin de comparer les segment (RFM) et le clustering.
        Parameters
        ----------
        X : pd.DataFrame ou np.array,
            Jeu de données d'apprentissage (normalisé). 
            
        dataframe : pd.DataFrame, 
            pandas DataFrame du jeu de données.
            
        segment_col : str, 
            Nom de la colonne contenant le nom des segement
            
        cluster_col : str, 
            Nom de la colonne contenant le nom les cluster
            
        Optional
        ----------        
        col_name_X : list, default=None
           Nom des colonnes si X est de type np.array.
            
        figsize : tulpe, default=(20,8)
            Indique les dimensions du graphique.
    """                
    if type(X) == type(pd.DataFrame()):
        X_scaled = X.copy()
    else:
        X_scaled = pd.DataFrame(X,columns=col_name_X)
    X_scaled["cluster"] = dataframe[cluster_col]
    X_scaled["segment"] = dataframe[segment_col]
    X_scaled["id"] = X_scaled.index
    test = pd.melt(frame= X_scaled,
                     id_vars= ['id', 'segment', 'cluster'],
                     var_name = 'metrics',
                     value_name = 'value')
    
    plt.style.use(style_plot)
    fig,axes=plt.subplots(1,2,figsize=figsize)

    sns.lineplot(x = 'metrics', y = 'value', hue = 'segment', data = test,ax=axes[0]) 
    sns.lineplot(x = 'metrics', y = 'value', hue = 'cluster', data = test,ax=axes[1]) 

    axes[0].set_title('Snake Plot of RFM')
    axes[1].set_title('Snake Plot of RFM')

    plt.show()

def plot_heatmap_comparatif(dataframe,
                            col_value,
                            segment_col=None,
                            cluster_col=None,
                            figsize=(20,8)):
    """Permet d'afficher un graphique sous forme de carte de chaleur du jeu de donnée afin de comparer les différente segmentation (cluster ou segment RFM) basé sur l'écart avec la moyenne total.
        Parameters
        ----------
        dataframe : pd.DataFrame, 
            pandas DataFrame du jeu de données d'apprentissage (non normalisé).
            
        Optional
        ----------
        segment_col : str, default=None, 
            Nom de la colonne contenant le nom des segement (mettre segment_col ou cluster_col ou les deux obligatoirement)
            
        cluster_col : str, default=None,
            Nom de la colonne contenant le nom les cluster (mettre segment_col ou cluster_col ou les deux obligatoirement)
            
        figsize : tulpe, default=(20,8)
            Indique les dimensions du graphique.
    """  
    plt.style.use(style_plot)
    heat = dataframe[col_value]
    if (not cluster_col in col_value) and ( not cluster_col is None):
        heat[cluster_col] = dataframe[cluster_col]
    
    if (not segment_col in col_value) and ( not segment_col is None):
        heat[segment_col] = dataframe[segment_col]
    
    # la valeur moyenne du total R F M
    total_avg = heat[col_value].mean() 

    # calcule l'écart proportionnel avec la moyenne totale 
    # RFM
    if not segment_col is None:
        cluster_avg = heat.groupby(segment_col)[col_value].mean()
        prop_rfm = cluster_avg/total_avg - 1
    
    # Kmeans
    if not cluster_col is None:
        cluster_avg_K = heat.groupby(cluster_col)[col_value].mean()
        prop_rfm_K = cluster_avg_K/total_avg - 1
        
    if (cluster_col is None) or (segment_col is None):
        fig,axes=plt.subplots(1,1,figsize=figsize)
    else:
        fig,axes=plt.subplots(1,2,figsize=figsize)


    # heatmap avec RFM 
    if (not segment_col is None) and (not cluster_col is None):
        sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True, ax=axes[0]) 
        axes[0].set_title('Heatmap of RFM quantile')
    elif cluster_col is None:
        sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True, ax=axes) 
        axes.set_title('Heatmap of RFM quantile')

    # heatmap avec K-means 
    if segment_col is None:
        sns.heatmap(prop_rfm_K, cmap= 'Blues', fmt= '.2f', annot = True, ax=axes)
        axes.set_title('Heatmap of K-Means')
    elif not cluster_col is None:
        sns.heatmap(prop_rfm_K, cmap= 'Blues', fmt= '.2f', annot = True, ax=axes[1]) 
        axes[1].set_title('Heatmap of K-Means')

    plt.show()
    

def plot_silhouette_sample(X,max_cluster=10):
    """Permet d'afficher une représentation des score silhouette pour chaque cluster grâce au 'silhouette_sample' de la méthode SilhouetteVisualizer de yellowbrick.
        Parameters
        ----------
        X : pd.DataFrame ou np.array, 
            pandas DataFrame du jeu de données d'apprentissage (normalisé).
            
        Optional
        ----------
        max_cluster : int, default=10
            Nombre max de cluster de 2 à max_cluster.
    """  
    for k in range(2,max_cluster):
        model = KMeans(k)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

        visualizer.fit(X)
        visualizer.show()
    
def plot_curve_davies_bouldin(X, max_cluster=10,
                              define_min_val=True,
                              figsize=(12,10),
                              max_iter=1000):
    """Permet d'afficher un graphique de la stabilité des cluster de 2 à max_cluster grâce au score de davies_bouldin (un résultat proche de 0 indique une stabilité importante = maximun d'inertie inter-cluster et minimum d'inertie intra-cluster)
        Parameters
        ----------
        X : pd.DataFrame ou np.array, 
            pandas DataFrame du jeu de données d'apprentissage (normalisé).
            
        Optional
        ----------
        max_cluster : int, default=10
            Nombre max de cluster de 2 à max_cluster.
            
        define_min_val : bool, default=True,
            Indique sur le graphique la valeur minimal de score avec écrit le cluster et la valeur de score
            
        figsize : tulpe, default=(12,10)
            Indique les dimensions du graphique.
            
        max_iter : int, default=1000
            Paramètre KMeans, permet de réduire le temps de calcul en limitant le nombre d'itération.
    """  
    plt.style.use(style_plot)
    score_davies_bouldin=[]
    for k in range(2,max_cluster):
        kmean = KMeans(n_clusters=k,max_iter=max_iter)
        kmean.fit(X)
        score_davies_bouldin.append(davies_bouldin_score(X, kmean.labels_))
    
    plt.figure(figsize=figsize)
    plt.title("Evolution score davies bouldin")
    plt.plot([i for i in range(2,max_cluster)],score_davies_bouldin,marker="o")
    
    if define_min_val:
        min_cluster = score_davies_bouldin.index(min(score_davies_bouldin)) + 2 # +2 car on commence à 2
        plt.plot([min_cluster,min_cluster],
                 [min(score_davies_bouldin),max(score_davies_bouldin)],
                 c="r",ls="--",label="min_value")
        plt.legend()
        print("Valeur minimal (score) :",round(min(score_davies_bouldin),3), "\nNombre de cluster :" + str(min_cluster))
    
    plt.xlabel("Nombres de clusters")
    plt.ylabel("Score (davies_bouldin)")
    plt.show()

def normalize_data(df,
                   plot_transformation=False,
                   rounded=3,
                   one_hot_encoder=None,
                   label_encoder=None,
                   log_col=False,
                   scaler=None):
    """Cette function permet de normaliser les donnée avec plusieur option possible comme un one_hot_encoder, label_encoder et la mise au log des variables, pour le scaler, par défaut l'algorithme utilise le StandardScaler de sklearn, il est possible d'ajouter un autre scaler de scikit-learn
        Parameters
        ----------
        df : pd.DataFrame, 
            pandas DataFrame du jeu de données d'apprentissage à transformer.
            
        Optional
        ----------
        plot_transformation : bool, default=False
            Indique si l'algorithme affiche la transformation des variables log effectuées.
            
        rounded : int, default=3,
            Arrondi les chiffres lors de la mise au log.
            
        one_hot_encoder : list, default=None
            Indique les colonnes où on applique l'encode one_hot_encoder.
            
        label_encoder : list, default=None
            Indique les colonnes où on applique l'encode label_encoder.
            
        log_col : bool, default=False
            Indique si on applique la mise au log des variables.
            
        scaler : Scaler (sklearn), default=None (StandardScaler)
            Permet de modifier le scaler par défaut (si None), il est possible de rajouter n'importe quelle Scaler de la librairie scikit-learn
                        
        Returns
        -------
        df_normalize, pd.DataFrame,
           Renvoie un DataFrame normalisé selon les critéres définies dans les paramètres.
    """
    cols = df.columns.tolist()
        
    df_normalize = df[cols].copy()
    all_object_encode = []
    
    if not one_hot_encoder is None:
        # Encode col
        all_object_encode.extend(one_hot_encoder)
        df_normalize = pd.get_dummies(df_normalize, columns=one_hot_encoder, dummy_na=False)
        all_new_col = [col for col in temp.columns if not col in cols]
        all_object_encode.extend(all_new_col)
    
    if not label_encoder is None:
        encoder_label = LabelEncoder()
        all_object_encode.extend(label_encoder)
        # Encode col et écrire les labels associés
        for label_col in label_encoder:
            print(label_col)
            df_normalize[label_col] = encoder_label.fit_transform(df_normalize[label_col]) 
            print({classe:index for index,classe in zip(range(len(encoder_label.classes_)),list(encoder_label.classes_))})
    
    
    col_not_object =  [col for col in cols if not col in all_object_encode]
    
    if log_col:
        # On enleve les valeurs négatives et les égales à 0 pour la mise au log
        df_normalize[col_not_object] = df_normalize[col_not_object].applymap(lambda x : 1 if x <= 0 else x)
        df_normalize[col_not_object] = df_normalize[col_not_object].apply(np.log,axis=1).round(rounded)

    if scaler is None:
        scaler = StandardScaler()
    
    X = np.abs(df_normalize[col_not_object])
    X_scaled = scaler.fit_transform(X)
    df_normalize[col_not_object] = X_scaled
    
    if plot_transformation:
        for col in col_not_object:
            fig, axes = plt.subplots(1,2,figsize=(20,5))
            sns.distplot(df[col],ax=axes[0])
            axes[0].set_title(col + " | Before")
            sns.distplot(df_normalize[col],ax=axes[1])
            axes[1].set_title(col + " | After")
            plt.show()
    
    return df_normalize


############ Satilité des clusters 

def prediction_multilabel(kmeans,X_test):
    """Cette function permet de prédire sur la base de la distance euclidien les centroids de l'algorithme KMeans (sklearn) le plus proche
        Parameters
        ----------
        kmeans : KMeans() (sklearn), 
            Algorithme de kmeans utilisé.
            
        X_test : pd.DataFrame, 
            pandas DataFrame du jeu de données à prédire.
            
        Returns
        -------
        prediction, np.array,
           Renvoie une liste numpy contenant les prédictions.    
    """
    if kmeans.cluster_centers_.shape[1] != X_test.shape[1]:
        Exception()
    return X_test.apply(lambda row : get_best_cluster(kmeans,row.values),axis=1).values

def calcul_distance(A,B):
    """Cette fonction calcul la distance entre 2 vecteur A et B (utilisé pour prediction_multilabel)
        Parameters
        ----------
        A : np.array, 
            Numpy array, vecteur A (individu à prédire).
            
        B : np.array, 
            Numpy array, vecteur B (centroid).
            
        Returns
        -------
        prediction, float,
           Renvoie la distance entre les 2 points.    
    """
    return np.sum(np.sqrt(((A-B)**2)),axis=0)

def get_best_cluster(kmeans,values):
    """Cette function cherche le meilleur cluster pour un point données
        Parameters
        ----------
        kmeans : KMeans() (sklearn), 
            Algorithme de kmeans utilisé.
            
        values : np.array, 
            Numpy array, vecteur individue (individu à prédire).
            
        Returns
        -------
        prediction, float,
           Renvoie l'index du meilleur cluster.    
    """
    minimum=999999999
    best_clust=0
    for cluster_index, centroid in zip(range(len(kmeans.cluster_centers_)),kmeans.cluster_centers_):
        dist = calcul_distance(values,centroid)
        if dist <= minimum:
            best_clust = cluster_index
            minimum = dist
    return best_clust

def stability_cluster_one_year(dataframe,
                               col_data,
                               random_state=42,
                               n_cluster=5,
                               recence_col="R",
                               plot_score_ari=True,
                               ceil_stability=0.85,
                               figsize=(9,7)):
    """Cette fonction permet de calculer la stabilité du clustering dans le temps (sur 1 an) cette fonction utilise un algorithme de KMeans par défaut. Le scoring est effectuer avec l'ARI (Adjuster Random Index) qui compare par pair les similarités de cluster (avec un score de 1 pour une ressemblance parfaite et négatif pour des clusters totalement différent). Par example : ARI([0,0,1,1],[1,1,0,0]) = 1.0
    
        Parameters
        ----------
        dataframe : pd.DataFrame, 
            pandas DataFrame du jeu de données d'apprentissage (non normalisé).
            
        col_data : list, 
            Liste des noms de colonnes.
            
        Optional
        ----------
        random_state : int, default=42
            Indique si l'algorithme affiche la transformation des variables log effectuées.
            
        n_cluster : int, default=5,
            Indique le nombre de cluster choisi pour l'algorithme de KMeans.
            
        recence_col : str, default='R'
            Indique le nom de la colonne de récence 'R' de 'RFM'
            
        plot_score_ari : bool, default=True
            Permet d'afficher un graphique des résultats pour un meilleur rendu
            
        ceil_stability : float, default=0.85
            Indique le seuil a afficher sur le graphique pour définir la stabilité du clustering dans le temps.
            
        figsize : tulpe, default=(9,7)
            Indique les dimensions du graphique.
                        
        Returns
        -------
        df_result, pd.DataFrame,
           Renvoie un DataFrame contenant les résultats.
    """
    all_freq_update = [30,90,180,270,360] # 1 mois, 3 mois, 6 mois, 9 mois, 1 ans
    last_train = 360 # on prend toutes les données avant 2017-08 pour train le modèle sois 1 ans

    # plus grande portion possible car plus petite section
    max_sample = dataframe[(dataframe[recence_col] <= last_train) & \
                             (dataframe[recence_col] > last_train - all_freq_update[0])].shape[0] 

    store_score_ari_test = []

    before = dataframe[dataframe[recence_col] > last_train]
    X_train = before[col_data]

    X_train = normalize_data(X_train,log_col=True)

    kmeans_before = KMeans(n_clusters=n_cluster)
    kmeans_before.fit(X_train)

    for freq in all_freq_update:
        # plage entre ]360 - freq ; 360]  (freq = 1,3,6,9,12 mois) 
        after = dataframe[(dataframe[recence_col] > last_train - freq)]

        X_test = after[col_data]
        X_test = normalize_data(X_test,log_col=True)

        kmeans_after = KMeans(n_clusters=n_cluster)
        kmeans_after.fit(X_test)

        # prédiction sur la distance des centroids
        y_true = prediction_multilabel(kmeans_before,X_test)
        y_pred = prediction_multilabel(kmeans_after,X_test)

        store_score_ari_test.append(adjusted_rand_score(y_true,y_pred))
        
    if plot_score_ari:
        plt.style.use(style_plot)
        plt.figure(figsize=figsize)
        plt.title("Stabilité des cluster dans le temps")
        plt.plot([i if i!=0 else 1  for i in range(0,13,3)],store_score_ari_test,marker="o")
        plt.plot([i if i!=0 else 1  for i in range(0,13,3)],
                 [ceil_stability for i in range(0,13,3)],
                 ls="--", c="r", label="Limite stabilité clustering")
        plt.legend()

        plt.xlabel("Nombre de mois")
        plt.ylabel("ARI-score")
        plt.show()
        
    return pd.DataFrame([
              ["1 mois","3 mois","6 mois","9 mois","1 ans"],
              store_score_ari_test
             ],
             index=[
                    "Fréquence",
                    "Ari_score"
                   ]).T

######## PCA 

def pca_transform(X,kmean=None,n_component=3,plot_corr_PCA=True,plot_data=True,figsize=(12,10),alpha_data=0.3):
    """Cette fonction permet de facilité la transformation PCA (Principal Composante Analysis) pour la projection des données X sur n_componnent, cela permet également d'afficher les corrélation des nouvelle composante avec le jeu de donnée initial et permet également de faire de la visualisation de donnée sur de plus petite dimension d'un jeu de données X.
    
        Parameters
        ----------
        X : pd.DataFrame, 
            pandas DataFrame du jeu de données d'apprentissage (normalisé).
            
        Optional
        ----------
        kmean : KMean (sklearn), default=None
            Mettre le kmean pour afficher les centroides lors de la présentation des données sur de plus petite dimension.
            
        n_component : int, default=5,
            Indique le nombre de composante principal a utiliser.
            
        plot_corr_PCA : str, default='R'
            Affiche la corrélation des composantes avec les donnée initial sous forme de heatmap
            
        plot_data : bool, default=True
            Permet d'afficher un graphique des données X sur les n_componnent (2 ou 3)
            
        alpha_data : float, default=0.3
            Indique la transparence des données affiché.
            
        figsize : tulpe, default=(12,10)
            Indique les dimensions du graphique.
                        
        Returns
        -------
        X_transform, pd.DataFrame,
           Renvoie un DataFrame contenant la projection sur le n_componnent avec le nomage ('PCA_X' X de 1 à n_componnent).
    """
    pca = PCA(n_components=n_component).fit(X)
    print("variance cumulé :",pca.explained_variance_ratio_.cumsum())
    component_name = ["PCA_" + str(i + 1) for i in range(n_component)]
    
    X_transform = pca.transform(X)
    X_ = pd.DataFrame(X_transform,columns=component_name)
    
    if not kmean is None:
        centroid = pca.transform(kmean.cluster_centers_)
        X_["cluster"] = kmean.labels_
    else:
        centroid=None
   
    
    if plot_corr_PCA:
        X_corr = X_.copy()
        for col in X.columns:
            X_corr[col] = X[col].values
        plt.figure(figsize=figsize)
        sns.heatmap(X_corr.corr()[component_name].iloc[n_component + 1:],annot=True)
        plt.show()
    
    if plot_data:
        plt.style.use(style_plot)
        if n_component>=3:
            fig = plt.figure(figsize=figsize)

            ax = fig.add_subplot(projection='3d')
            if not kmean is None:
                # Centroid
                ax.scatter(centroid[:,0],centroid[:,1],centroid[:,2],c="red",s=100,alpha=1,label="Centroid")
                # legend centroid
                legend = ax.legend(loc="lower right")
                ax.add_artist(legend)
            
            # donnée par cluster 
            if not kmean is None:
                scatter = ax.scatter(X_["PCA_1"], X_["PCA_2"], X_["PCA_3"], c=X_["cluster"],cmap="viridis",alpha=alpha_data)
            else:
                ax.scatter(X_["PCA_1"], X_["PCA_2"], X_["PCA_3"],alpha=alpha_data)
            
            # legend des cluster
            if not kmean is None:
                handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
                legend_1 = ax.legend(handles, labels, loc="upper right", title="Cluster")
            
            # renomage des axis
            ax.set_xlabel("PCA_1")
            ax.set_ylabel("PCA_2")
            ax.set_zlabel("PCA_3")

            plt.show()
        elif n_component==2:
            fig = plt.figure(figsize=figsize)

            ax = fig.add_subplot()            
             
            if not kmean is None:
                # donnée par cluster
                scatter = ax.scatter(X_["PCA_1"], X_["PCA_2"], c=X_["cluster"],cmap="viridis",alpha=alpha_data)
                # legend des cluster
                handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
                legend_1 = ax.legend(handles, labels, loc="upper right", title="Cluster")
                ax.add_artist(legend_1)
            else:
                ax.scatter(X_["PCA_1"], X_["PCA_2"],alpha=alpha_data)
            
            if not kmean is None:
                # Centroid
                ax.scatter(centroid[:,0],centroid[:,1],c="red",s=100,label="Centroid")
                # legend centroid
                legend = ax.legend(loc="lower right")
            
            # renomage des axis
            ax.set_xlabel("PCA_1")
            ax.set_ylabel("PCA_2")

            plt.show()
            
            
    return X_[component_name]


def all_plan(start,n_comp=4,just_next_plan=True):
    """Cette fonction permet de générer une liste des plan a utilisé dans la fonction 'display_circles'. Par exemple avec start à 0, n_comp = 4 et just_next_plan = False => ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)) et avec start à 0, n_comp = 4 et just_next_plan = True => ((0,1),(2,3)) sois les composante '((PCA_1 et PCA_2),(PCA_3 et PCA_4))
    
        Parameters
        ----------
        start : pd.DataFrame, 
            pandas DataFrame du jeu de données d'apprentissage (non normalisé).
            
        Optional
        ----------
        n_comp : int, default=4
            Nombre de composante utilisé dans le PCA.
            
        just_next_plan : bool, default=True,
            Indique si on utilise uniquement les plans par pairs
                        
        Returns
        -------
        df_result, list,
           Renvoie la liste des plan.
    """
    plan=[]
    if just_next_plan:
        for i in range(start+1,n_comp,2):
            plan.append((i-1,i))
    else:
        for i in range(start,n_comp):
            for j in range(i + 1,n_comp):
                plan.append((i,j))
    return plan

def display_circles(pcs,pca, axis_ranks,n_comp=4, labels=None, label_rotation=0, lims=None):
    """Permet d'afficher le cercle des corrélation du PCA sur les plan définis.
    
        Parameters
        ----------
        pcs : ..., 
            ...
            
        pca : ..., 
            ...
            
        axis_ranks : ..., 
            ...
            
        Optional
        ----------
        n_comp : int, default=4
            ...
            
        labels : list, default=None,
            ...
            
        label_rotation : int, default=0
            ...
            
        lims : list, default=None
            ...
    """
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10,9))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)