

IndexLoader [get the indices of the (partial) dataset]
 | -- get_all_idx
 | -- get_idx return target idx
 | -- get_folds_idx **return** fold train idx, fold valid idx


ScaleLoader [Summarize the counts on the (partial) dataset, save scaler and data transformation]
 | -- load_scaler
       | -- 0.create_scaler
          || -- IndexLoader
       | -- 1.save_scaler
 | -- scale_samples


DatasetLoader [Load dataset and get (all kinds of) informatin from the (partial) dataset]
 | -- init_count_pd
 | -- get_count_pd
 | -- get_base_dir
 | -- get_undersample_folder_name
 | -- get_feature_number
 | -- 0.init_empty_dataset
       | -- init_target_id_and_file_name
          || -- IndexLoader
 | -- 1.load_data


-- need to summarize later
Trainer [Train the model given a DatasetLoader]
 | -- load_scaler
    ||-- ScaleLoader
 | -- load_data (Load and scale dataset)
    ||-- DatasetLoader
    ||-- ScaleLoader
 | -- load_model
 | -- save_model
 | -- train
 | -- test_patient_wise

