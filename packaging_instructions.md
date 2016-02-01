To build a new version, here are the steps to follow:

1. If necessary, download new data with `python PyWPA/preprocess.py`.
2. If necessary, rebuild the model with `python PyWPA/model.py`.
3. If necessary, zip the data with `tar czf PyWPA/data/nfldb_processed_data.tar.gz PyWPA/data/nfldb_processed_data_*csv`.
4. If necessary, zip the model with `tar czf PyWPA/models/PyWPA_model.tar.gz PyWPA/models/PyWPA_model.joblib.pkl*`.
