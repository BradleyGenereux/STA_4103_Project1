import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_file = ["LogReg.pkl","SVC.pkl","DecTree.pkl","RanFor.pkl","NaiBayes.pkl","KNear.pkl"]

# user must input given parameters to predict if this house would be affordable for the average household in miami
latitude = 25.9         # LATITUDE
longitude = -80.1       # LONGITUDE
lnd_sqfoot = 10000      # LND_SQFOOT: land area (square feet)
tot_lvg_area = 1500     # TOT_LVG_AREA: floor area (square feet)
spec_feat_val = 3500    # SPEC_FEAT_VAL: value of special features (e.g., swimming pools) ($)
rail_dist = 2000        # RAIL_DIST: distance to the nearest rail line (an indicator of noise) (feet)
ocean_dist = 15000      # OCEAN_DIST: distance to the ocean (feet)
water_dist = 50         # WATER_DIST: distance to the nearest body of water (feet)
cntr_dist = 45000       # CNTR_DIST: distance to the Miami central business district (feet)
subcntr_dist = 45000    # SUBCNTR_DI: distance to the nearest subcenter (feet)
hwy_dist = 20000        # HWY_DIST: distance to the nearest highway (an indicator of noise) (feet)
age = 50                # age: age of the structure
avno60plus = 0          # avno60plus: dummy variable for airplane noise exceeding an acceptable level     [0,1]
structure_quality = 4   # structure_quality: quality of the structure                                     [1,2,3,4,5]

# create array from user input data
x = [[latitude, longitude, lnd_sqfoot, tot_lvg_area, spec_feat_val, rail_dist, ocean_dist, water_dist, cntr_dist, subcntr_dist, hwy_dist, age, avno60plus, structure_quality]]

ss_test = StandardScaler()
x_test = ss_test.fit_transform(x)

# create prediciton from user data
prediction = []
for i in range(0,len(model_file)):
    a = joblib.load(model_file[i]).predict(x_test)
    if a == 0: prediction.append("Unaffordable")
    else: prediction.append("Affordable")

# print predictions
df_model = pd.DataFrame(index=model_file, columns=['Prediction'])
df_model['Prediction'] = prediction
print(df_model)