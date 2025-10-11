import yaml
import os

class ModelConfig:
    YML_FILE_PATH = "configs/models.yaml"
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.yml_file_path = os.path.join(base_dir, self.YML_FILE_PATH)
        self.models = self.load()

    def load(self):
        print(f"file path = {self.yml_file_path}")
        with open(self.yml_file_path, 'r') as file:
            models = yaml.safe_load(file).get('models', [])
        return models
    
    def __get_model_attr(self, attr):
        return [model[attr] for model in self.models]
    
    def __get_attr_by_model_id(self, model_id, attr):
        return next(model[attr] for model in self.models if model['id'] == model_id)
    
    def get_model_ids(self):
        return self.__get_model_attr('id')
    
    def get_model_names(self):
        return self.__get_model_attr('name')
    
    def is_model_available(self, model_id):
        return model_id in self.get_model_ids()
    
    def get_model_type(self, model_id):
        return self.__get_attr_by_model_id(model_id, "model_type")
    
    def get_hidden_layers(self, model_id):
        return self.__get_attr_by_model_id(model_id, "hidden_layers")

    
    


