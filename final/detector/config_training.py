config = {'train_data_path':['../data/train/'],
          'val_data_path':['../data/test/'], 
          'test_data_path':['../data/test/'], 
          
          'train_preprocess_result_path':'../data/prep/', # contains numpy for the data and label, which is generated by prepare.py
          'val_preprocess_result_path':'../data/prep/',  # make sure copy all the numpy into one folder after prepare.py
          'test_preprocess_result_path':'../data/prep/',
          
          'train_annos_path':'../data/annotations.csv',
          'val_annos_path':'../data/annotations.csv',
          'test_annos_path':'../data/annotations.csv',
          
          'black_list':[],
          
          'preprocessing_backend':'python3',

          'luna_segment':'../data/seg/pall/', # download from seg-lungs-LUNA16/seg-lungs-LUNA16/
          'preprocess_result_path':'../data/prep/',
          'luna_data':'../data/',
          'luna_label':'../data/annotations.csv'
         } 