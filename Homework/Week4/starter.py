#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--year", type=int,
                    help="insert the year")
parser.add_argument("--month", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    help="insert the month")
args = parser.parse_args()



with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def predict(year: int, month: int):
    

    #df = read_data(f'/home/aparicio/MLOps_Zoomcamp/data/yellow_trip_data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("Average predicted duration:", y_pred.mean())


if __name__ == "__main__":
    year = args.year
    month = args.month
    predict(year, month)

