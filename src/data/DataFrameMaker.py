import pandas as pd
import scipy.io
from pathlib import Path
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


class DataFrameMaker:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.images_folder = self.base_path / 'image'
        self.labels_folder = self.base_path / 'label'
        self.misc_folder = self.base_path / 'misc'
        self.attributes_df = None
        self.make_mapping = None
        self.model_mapping = None
        self.type_mapping = None
        self.dataset_df = None
        
        logging.info(f"Initialized DataFrame maker for {self.base_path} folder")

    def load_metadata(self):
        """Loads the dataset metadata and create it's colunmns"""
        logging.info("Loading metadata into to the maker")
        self.attributes_df = pd.read_csv(self.misc_folder / 'attributes.txt', sep=' ')
        self.attributes_df['model_id'] = self.attributes_df['model_id'].astype(str)
        
        make_model_mat = scipy.io.loadmat(self.misc_folder / 'make_model_name.mat')
        car_type_mat = scipy.io.loadmat(self.misc_folder / 'car_type.mat')

        make_names = [str(x[0][0]) for x in make_model_mat['make_names']]
        model_names = [str(x[0][0]) if len(x) > 0 and len(x[0]) > 0 else '' for x in make_model_mat['model_names']]
        car_types = [str(x[0]) for x in car_type_mat['types'][0]]
        
        self.make_mapping = {str(i+1): name for i, name in enumerate(make_names)}
        self.model_mapping = {str(i+1): name for i, name in enumerate(model_names)}
        self.type_mapping = {str(i+1): name for i, name in enumerate(car_types)}
        self.type_mapping['0'] = 'Unknown'
        logging.info("Metadata loaded in the maker")

    def process_image_data(self, image_path, viewpoint_filter={1, 2}):
        """Process the images data and create the columns for them"""
        try:
            logging.info("Processing images")
            label_path = str(image_path).replace(str(self.images_folder), str(self.labels_folder)).replace('.jpg', '.txt')
            relative_path = image_path.relative_to(self.images_folder)
            make_id, model_id, released_year, image_name = relative_path.parts
            
            with open(label_path, 'r') as f:
                viewpoint = int(f.readline().strip())
                f.readline()  
                bb_coords = f.readline().strip()
            
            if viewpoint not in viewpoint_filter:
                return None
                
            x1, y1, x2, y2 = map(float, bb_coords.split())
            logging.info("Images processed with succes")

            return {
                'image_name': image_name.replace('.jpg', ''),
                'image_path': str(image_path),
                'make_id': make_id,
                'model_id': model_id,
                'released_year': released_year,
                'viewpoint': viewpoint,
                'bbox': [x1, y1, x2, y2]
                }
        except Exception as e:
            logging.warning(f"Error procesando {image_path}: {e}")
            return None

    def build_dataset(self):
        """Builds the dataset by calling the other methods"""
        if self.attributes_df is None:
            self.load_metadata()
        
        dataset_list = []
        for image_path in self.images_folder.rglob('*.jpg'):
            data = self.process_image_data(image_path)
            if data:
                dataset_list.append(data)
        
        if not dataset_list:
            logging.info("No images found")
            return None
        
        self.dataset_df = pd.DataFrame(dataset_list)
        viewpoint_mapping = {1: 'front', 2: 'rear'}
        self.dataset_df['viewpoint'] = self.dataset_df['viewpoint'].map(viewpoint_mapping)
        self.dataset_df = self.dataset_df.merge(self.attributes_df, on='model_id', how='left')
        self.dataset_df['make'] = self.dataset_df['make_id'].map(self.make_mapping)
        self.dataset_df['model'] = self.dataset_df['model_id'].map(self.model_mapping)
        self.dataset_df['type'] = self.dataset_df['type'].astype(int).astype(str).map(self.type_mapping)
        
        columns_to_keep = [
            'image_name', 'image_path', 'released_year', 'viewpoint',
            'bbox', 'make', 'model', 'type'
            ]
        self.dataset_df = self.dataset_df[columns_to_keep]
        logging.info(f"Final dataset with: {len(self.dataset_df)} images")
        logging.info(f"View points distribution: {self.dataset_df['viewpoint'].value_counts().to_dict()}")
        return self.dataset_df