import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, Model, CharField, IntegerField, FloatField,
    IntegrityError, CompositeKey
)

# 1. Database setup
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.environ.get('DATABASE_URL') or os.path.join(BASE_DIR, 'forecasts.db')
db = SqliteDatabase(DB_PATH)

class Forecast(Model):
    sku = CharField(max_length=50)
    time_key = IntegerField()
    pvp_is_competitorA = FloatField()
    pvp_is_competitorB = FloatField()
    pvp_is_competitorA_actual = FloatField(null=True)
    pvp_is_competitorB_actual = FloatField(null=True)

    class Meta:
        database = db
        primary_key = CompositeKey('sku', 'time_key')

Forecast.create_table(safe=True)

# 2. Paths to data and pipelines
CAMPAIGNS_CSV = os.path.join(BASE_DIR, 'train', 'chain_campaigns.csv')
PRICES_CSV    = os.path.join(BASE_DIR, 'train', 'product_prices_leaflets.csv')
STRUCT_CSV    = os.path.join(BASE_DIR, 'train', 'product_structures_sales.csv')
PIPE_A_PKL    = os.path.join(BASE_DIR, 'pipeline_A_1.pickle')
PIPE_B_PKL    = os.path.join(BASE_DIR, 'pipeline_B_1.pickle')

# 3. Load and prepare feature dataframe
def load_features():
    # Campaigns
    df1 = pd.read_csv(CAMPAIGNS_CSV)
    df1['start_date'] = pd.to_datetime(df1['start_date'])
    df1['end_date']   = pd.to_datetime(df1['end_date'])
    df1['duration_days'] = (df1['end_date'] - df1['start_date']).dt.days + 1

    # Prices
    df2 = pd.read_csv(PRICES_CSV)
    df2['time_key'] = pd.to_datetime(df2['time_key'].astype(str), format='%Y%m%d')
    df2 = df2[df2['discount'] >= 0].copy()
    df2['effective_price'] = df2['pvp_was'] * (1 - df2['discount'])

    # Structure & sales
    df3 = pd.read_csv(STRUCT_CSV)
    df3['time_key'] = pd.to_datetime(df3['time_key'].astype(str), format='%Y%m%d')
    df3 = df3[df3['quantity'] >= 0].copy()
    for lvl in ['structure_level_1','structure_level_2','structure_level_3','structure_level_4']:
        df3[lvl] = df3[lvl].astype('category')

    # Merge chain, A, B
    df_merged = pd.merge(df2, df3, on=['sku','time_key'], how='left')
    # Chain
    df_chain = df_merged[df_merged['competitor']=='chain'].rename(columns={
        'pvp_was':'pvp_was_chain','discount':'discount_chain',
        'flag_promo':'flag_promo_chain','quantity':'quantity_chain'
    }).drop(columns=['competitor','effective_price'])
    # A
    df_A = df_merged[df_merged['competitor']=='competitorA'].rename(columns={
        'pvp_was':'pvp_was_A','discount':'discount_A',
        'flag_promo':'flag_promo_A','quantity':'quantity_A',
        'effective_price':'effective_price_A'
    }).drop(columns=['competitor'])
    # B
    df_B = df_merged[df_merged['competitor']=='competitorB'].rename(columns={
        'pvp_was':'pvp_was_B','discount':'discount_B',
        'flag_promo':'flag_promo_B','quantity':'quantity_B',
        'effective_price':'effective_price_B'
    }).drop(columns=['competitor'])

    # Merge all three
    df_wide = pd.merge(df_chain, df_A, on=['sku','time_key'], how='left')
    df_wide = pd.merge(df_wide, df_B, on=['sku','time_key'], how='left')

    # Normalize SKU
    df_wide['sku'] = df_wide['sku'].astype(str)

    # Time features
    df_wide['month']        = df_wide['time_key'].dt.month
    df_wide['day_of_week']  = df_wide['time_key'].dt.weekday
    df_wide['day_of_month'] = df_wide['time_key'].dt.day
    df_wide['year']         = df_wide['time_key'].dt.year

    # Sort for lag
    df_wide = df_wide.sort_values(['sku','time_key']).reset_index(drop=True)

    # Lag features A
    df_wide['price_lag_1_A'] = df_wide.groupby('sku')['pvp_was_A'].shift(1)
    df_wide['price_roll_mean_7_A'] = (
        df_wide.groupby('sku')['pvp_was_A'].shift(1)
                .rolling(7).mean().reset_index(level=0, drop=True)
    )
    # Lag features B
    df_wide['price_lag_1_B'] = df_wide.groupby('sku')['pvp_was_B'].shift(1)
    df_wide['price_roll_mean_7_B'] = (
        df_wide.groupby('sku')['pvp_was_B'].shift(1)
                .rolling(7).mean().reset_index(level=0, drop=True)
    )

    # Compute spreads A & B
    df_A_warm = df_wide.dropna(subset=['price_lag_1_A'])
    df_A_warm['spread_A'] = df_A_warm['price_lag_1_A'] - df_A_warm['pvp_was_chain']
    cat_spread_A = df_A_warm.groupby('structure_level_2')['spread_A'].mean().to_dict()
    global_spread_A = df_A_warm['spread_A'].mean()

    df_B_warm = df_wide.dropna(subset=['price_lag_1_B'])
    df_B_warm['spread_B'] = df_B_warm['price_lag_1_B'] - df_B_warm['pvp_was_chain']
    cat_spread_B = df_B_warm.groupby('structure_level_2')['spread_B'].mean().to_dict()
    global_spread_B = df_B_warm['spread_B'].mean()

    return df_wide, cat_spread_A, global_spread_A, cat_spread_B, global_spread_B

# Load once
DF, CAT_SP_A, GLOB_SP_A, CAT_SP_B, GLOB_SP_B = load_features()

# Define feature order for prediction
FEATURE_COLS = [
    'sku',
    'pvp_was_chain','discount_chain','flag_promo_chain',
    'structure_level_1','structure_level_2','structure_level_3','structure_level_4',
    'quantity_chain','flag_promo_A','flag_promo_B',
    'month','day_of_week','day_of_month','year',
    'price_lag_1_A','price_roll_mean_7_A','price_lag_1_B','price_roll_mean_7_B',
    'is_imputed_A','is_imputed_B'
]

# 4. Load pipelines
pipeline_A = joblib.load(PIPE_A_PKL)
pipeline_B = joblib.load(PIPE_B_PKL)

# Helper impute functions
def impute_A(row):
    sp = CAT_SP_A.get(row['structure_level_2'], GLOB_SP_A)
    base = row['pvp_was_chain'] + sp
    return base, base

def impute_B(row):
    sp = CAT_SP_B.get(row['structure_level_2'], GLOB_SP_B)
    base = row['pvp_was_chain'] + sp
    return base, base

# Build features for a single request
from flask import Flask
app = Flask(__name__)

def build_features(sku, time_key):
    tk = pd.to_datetime(str(time_key), format='%Y%m%d', errors='coerce')
    if pd.isna(tk): return None
    df_row = DF[(DF['sku']==str(sku)) & (DF['time_key']==tk)]
    if df_row.empty: return None
    row = df_row.iloc[0].copy()
    row['sku'] = sku  # ensure sku is present
    # A impute
    if pd.isna(row['price_lag_1_A']):
        row['price_lag_1_A'], row['price_roll_mean_7_A'] = impute_A(row)
        row['is_imputed_A'] = 1
    else:
        row['is_imputed_A'] = 0
    # B impute
    if pd.isna(row['price_lag_1_B']):
        row['price_lag_1_B'], row['price_roll_mean_7_B'] = impute_B(row)
        row['is_imputed_B'] = 1
    else:
        row['is_imputed_B'] = 0
    return pd.DataFrame([row[FEATURE_COLS]])

# 5. Routes
def make_app():
    @app.route('/forecast_prices/', methods=['POST'])
    def forecast_prices():
        req = request.get_json() or {}
        sku = req.get('sku'); time_key = req.get('time_key')
        if not isinstance(sku, str) or not isinstance(time_key, int):
            return jsonify({'error':'Invalid input'}),422
        X = build_features(sku, time_key)
        if X is None:
            return jsonify({'error':'SKU or date not found'}),422
        try:
            pA = float(pipeline_A.predict(X)[0])
            pB = float(pipeline_B.predict(X)[0])
        except Exception as e:
            return jsonify({'error':f'Prediction error: {e}'}),500
        try:
            Forecast.create(
                sku=sku, time_key=time_key,
                pvp_is_competitorA=pA, pvp_is_competitorB=pB
            )
        except IntegrityError:
            return jsonify({'error':'Forecast exists'}),422
        return jsonify({'sku':sku,'time_key':time_key,
                        'pvp_is_competitorA':pA,'pvp_is_competitorB':pB}),200
    
    @app.route('/actual_prices/', methods=['POST'])
    def actual_prices():
        req = request.get_json() or {}
        sku = req.get('sku'); time_key=req.get('time_key')
        aA = req.get('pvp_is_competitorA_actual'); aB=req.get('pvp_is_competitorB_actual')
        if not all([isinstance(sku,str),isinstance(time_key,int),
                    isinstance(aA,(int,float)), isinstance(aB,(int,float))]):
            return jsonify({'error':'Invalid input'}),422
        try:
            rec = Forecast.get(Forecast.sku==sku, Forecast.time_key==time_key)
        except Forecast.DoesNotExist:
            return jsonify({'error':'No forecast'}),422
        rec.pvp_is_competitorA_actual = aA
        rec.pvp_is_competitorB_actual = aB
        rec.save()
        return jsonify({'sku':sku,'time_key':time_key,
                        'pvp_is_competitorA':rec.pvp_is_competitorA,
                        'pvp_is_competitorB':rec.pvp_is_competitorB,
                        'pvp_is_competitorA_actual':aA,
                        'pvp_is_competitorB_actual':aB}),200

    return app

if __name__=='__main__':
    make_app().run(host='0.0.0.0', port=5000, debug=True)
