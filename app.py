import streamlit as st
import joblib
from sklearn.tree import DecisionTreeRegressor
import pandas as pd


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
product_names = ['Produit A', 'Produit B', 'Produit C']
categories = ['Catégorie 1', 'Catégorie 2']
manufacturers = ['Fab A', 'Fab B']
villes = ['Douala', 'Yaoundé','Limbe','Bafoussam','Dschang','Bertouai']
units = [1, 2, 5, 10]

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
        unit = st.selectbox("🔢 Quantité par commande", units)

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