import streamlit as st
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# bibliotheque
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline

def preprocessing(df):
    print('preprocessing')
    cat_features = ['ProductName','Categorie','manufacturer','Ville']
    num_features = ['Unit']
    num_tranformer = Pipeline([('scaler',MinMaxScaler())])
    cat_transformer = Pipeline([('encoder',OneHotEncoder(handle_unknown = 'ignore',sparse_output=False))])
    preprocessor = ColumnTransformer([
        ('num',num_tranformer,num_features),
        ('cat',cat_transformer,cat_features)
    ])
    df_clean = preprocessor.fit_transform(df)
    if hasattr(df_clean,'toarray'):
       df_clean = df_clean.toarray()
    new_columns = (
        num_features +
        list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features))
    )
    df_clean = pd.DataFrame(df_clean,columns=new_columns)
    df_clean['Mois'] = df.Mois
    df_clean['Jour'] = df.Jour
    df_clean['Annee'] = df.Annee
    return df_clean
    


# Chargement du modèle
model = joblib.load("modele.joblib")

# Listes de choix pour la saisie manuelle
product_names = ['Scotch Tape 29','Converse All Star 16','Barbie Fashion Set 9',
 'H&M Dress 5','Zara Coat 37','Pilot G2 Pen 28','Post-it Notes 40',
 'Adidas Superstar 11','Converse All Star 20','Coca-Cola Can 23',
 'Nike T-Shirt 25','Pilot G2 Pen 34','Coca-Cola Can 4',
 'Playmobil Pirate Set 10','Post-it Notes 14','Adidas Shorts 4',
 'Fisher-Price Giggles Set 27','Post-it Notes 26','Oreo Cookies 24',
 'Oreo Cookies 6','Adidas Superstar 3' "Lay's Chips 12" 'Nutella Jar 39',
 'Nike Air Max 19','Staples Pack 8','Nutella Jar 21' "Lay's Chips 25",
 'New Balance 574 24','Lego City Set 2','Uniqlo Jeans 33',
 'Post-it Notes 13','Zara Coat 16','Hot Wheels Set 29',
 'Samsung Galaxy Model 110','Playmobil Pirate Set 35',
 'Barbie Fashion Set 17','Oreo Cookies 5','Pilot G2 Pen 6',
 'Uniqlo Jeans 18','Scotch Tape 18','Adidas Superstar 6',
 'Post-it Notes 32','Kinder Surprise 9','Oxford Notebook 2','H&M Dress 15',
 'Playmobil Pirate Set 33','Oxford Notebook 22','Nike T-Shirt 39',
 'Nike Air Max 13','Lego City Set 21','Staples Pack 3' "Lay's Chips 2",
 'Pilot G2 Pen 38','Barbie Fashion Set 19','Adidas Superstar 9',
 'Kinder Surprise 33','Nike T-Shirt 20','Nutella Jar 28','Puma Running 8',
 "Lay's Chips 38" 'Apple iPhone Model 120','Adidas Shorts 13',
 'Fisher-Price Giggles Set 34','Kinder Surprise 34','Pilot G2 Pen 36',
 'Pilot G2 Pen 27','Puma Running 23','H&M Dress 36','Lego City Set 22',
 'Pilot G2 Pen 16','Adidas Shorts 21','Nike T-Shirt 31','Zara Coat 40',
 'Zara Coat 6','Lego City Set 31','Adidas Shorts 23',
 'Converse All Star 10','Kinder Surprise 22','Nike T-Shirt 1',
 'Oxford Notebook 23','Kinder Surprise 29','H&M Dress 14',
 'Staples Pack 11','Oreo Cookies 18','H&M Dress 32','Oreo Cookies 7',
 'Playmobil Pirate Set 13','Uniqlo Jeans 8','Oreo Cookies 14',
 'Barbie Fashion Set 1','Puma Running 5','Adidas Shorts 12',
 'Playmobil Pirate Set 8','Nutella Jar 36','Uniqlo Jeans 27','Converse All Star 22' "Lay's Chips 27" 'Post-it Notes 10',
 'Oxford Notebook 25','Lego City Set 26','Oxford Notebook 9','Apple iPhone Model 139','Apple iPhone Model 103','Nutella Jar 1',
 'Fisher-Price Giggles Set 24','Nutella Jar 13','Uniqlo Jeans 10',
 'Scotch Tape 20' "Lay's Chips 31" "Lay's Chips 20" 'Hot Wheels Set 16',
 'Puma Running 17','Adidas Shorts 24','Nutella Jar 15','Uniqlo Jeans 17',
 'Huawei P Model 142','Adidas Shorts 28','Pilot G2 Pen 24',
 'Pilot G2 Pen 33','Oxford Notebook 5','Adidas Superstar 4',
 'Adidas Superstar 15' "Lay's Chips 8" 'Puma Running 7',
 'New Balance 574 21','Playmobil Pirate Set 14','Playmobil Pirate Set 40',
 'Coca-Cola Can 10','H&M Dress 29','Lego City Set 18','Hot Wheels Set 3',
 'Oxford Notebook 30','Samsung Galaxy Model 106',
 'Fisher-Price Giggles Set 15','Sony Xperia Model 135',
 'Playmobil Pirate Set 39','Nutella Jar 30','Kinder Surprise 11',
 'Coca-Cola Can 32','H&M Dress 3','New Balance 574 18','H&M Dress 9',
 'Nike Air Max 1','Zara Coat 26','Adidas Shorts 7','Oxford Notebook 17',
 'Oxford Notebook 4','Playmobil Pirate Set 38','Playmobil Pirate Set 36',
 'Kinder Surprise 17','Staples Pack 12','Huawei P Model 117' "Lay's Chips 26" 'Adidas Superstar 12','Nike T-Shirt 2',
 'New Balance 574 14','Scotch Tape 7','Hot Wheels Set 30',
 'Playmobil Pirate Set 7','Nike T-Shirt 30','Playmobil Pirate Set 23',
 'Adidas Shorts 11','Playmobil Pirate Set 20','Nutella Jar 35',
 'Hot Wheels Set 25','Staples Pack 15','Zara Coat 35',
 'Samsung Galaxy Model 149','Samsung Galaxy Model 147','Oreo Cookies 40',
 'Samsung Galaxy Model 108','H&M Dress 19','Lego City Set 6','Pilot G2 Pen 39','Nutella Jar 16' "Lay's Chips 3" 'New Balance 574 2',
 'Samsung Galaxy Model 144','Pilot G2 Pen 35','Pilot G2 Pen 37',
 'Barbie Fashion Set 4','Huawei P Model 118','Staples Pack 21',
 'Kinder Surprise 37','Lego City Set 37','Playmobil Pirate Set 32',
 'Nike T-Shirt 38','Oxford Notebook 19','Samsung Galaxy Model 146',
 'Post-it Notes 1','Sony Xperia Model 107','Sony Xperia Model 140',
 'Playmobil Pirate Set 12','LG Velvet Model 136','Barbie Fashion Set 11',
 'Sony Xperia Model 128','Adidas Shorts 22','Sony Xperia Model 137',
 'Lego City Set 28','Sony Xperia Model 111','Playmobil Pirate Set 5',
 'Sony Xperia Model 125','Sony Xperia Model 114','Apple iPhone Model 130',
 'Pilot G2 Pen 31','Samsung Galaxy Model 100','Kinder Surprise 19',
 'Apple iPhone Model 134','Apple iPhone Model 102',
 'Apple iPhone Model 141','H&M Dress 34','Huawei P Model 119',
 'Sony Xperia Model 105','Sony Xperia Model 104',
 'Samsung Galaxy Model 122','Sony Xperia Model 123','Huawei P Model 133',
 'Samsung Galaxy Model 138','Sony Xperia Model 112','Huawei P Model 131',
 'Samsung Galaxy Model 115','Apple iPhone Model 143','LG Velvet Model 148',
 'LG Velvet Model 127','Sony Xperia Model 116','Apple iPhone Model 113',
 'Sony Xperia Model 126','LG Velvet Model 145','Sony Xperia Model 129',
 'Apple iPhone Model 121','Samsung Galaxy Model 101',
 'Samsung Galaxy Model 109','Samsung Galaxy Model 124',
 'Apple iPhone Model 132']
categories = ['Fournitures','Chaussures','Jouets','Vêtements','Nourriture','Électronique']
manufacturers = ['Nike','Johnson & Johnson','Nestlé','Sony','Unilever','LG','Samsung','Procter & Gamble','Heineken',"L'Oréal",'Danone','Apple','Panasonic','Microsoft','Adidas','Huawei','Philips']
villes = ['Douala', 'Yaoundé','Limbe','Bafoussam','Dschang','Bertouai']


# Titre
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🛒 Prédiction de Stock Intelligent</h1>", unsafe_allow_html=True)
st.markdown("---")

# Choix du mode
mode = st.radio("Choisissez le mode de prédiction :", ["📝 Saisie manuelle", "📁 Importer un fichier (.csv ou .xlsx)"])

# -------------------------
# MODE 1 : SAISIE MANUELLE
# -------------------------
if mode == "📝 Saisie manuelle":
    st.markdown("### Veuillez renseigner les informations du produit :")
    col1, col2 = st.columns(2)

    with col1:
        product = st.selectbox("📦 Nom du produit", product_names)
        manufacturer = st.selectbox("🏭 Fabricant", manufacturers)
        unit = st.number_input("Prix unitaire", format="%.2f")
    with col2:
        categorie = st.selectbox("🗂️ Catégorie", categories)
        ville = st.selectbox("🌍 Ville", villes)

    st.markdown("---")
    if st.button("🔍 Prédire (manuel)"):
        input_df = pd.DataFrame([{
            'ProductName': product,
            'Categorie': categorie,
            'manufacturer': manufacturer,
            'Ville': ville,
            'Unit': unit
        }])

        input_processed = preprocessing(input_df)
        prediction = model.predict(input_processed)[0]

        st.markdown(f"""
        <div style='text-align: center; font-size: 24px; color: #2196F3; padding: 20px; background-color: #E3F2FD; border-radius: 10px;'>
            ✅ Quantité prédite : <strong>{prediction:.2f} unités</strong>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# MODE 2 : FICHIER CSV / EXCEL
# -------------------------
else:
    uploaded_file = st.file_uploader("Importer un fichier (.csv ou .xlsx)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"{len(df)} lignes chargées avec succès.")
            st.write("Aperçu des données :", df.head())

            if st.button("🔍 Prédire (fichier)"):
                df_encoded = preprocessing(df.copy())
                predictions = model.predict(df_encoded)
                df['Quantite_Predite'] = predictions

                # Télécharger le fichier avec prédictions
                csv_result = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger le fichier avec prédictions",
                    data=csv_result,
                    file_name='predictions.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
