import yaml
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
class Config():
    def __init__(self, yaml_file=os.path.join(current_directory, 'general_parameters.yaml')):
        # print(current_directory)
        # print(os.path.join(current_directory, 'general_parameters.yaml'))
        # # print(os.path.dirname(os.path.realpath(__file__)))
        # print(os.getcwd())
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)

            self.dataset_folder= config.get('dataset_folder')
            self.cuda_device = config.get('cuda_device')
            self.save_visual_results = config.get('save_visual_results')

            self.quantization_size = config.get('quantization_size')
            self.num_workers = config.get('num_workers')
            self.batch_size = config.get('batch_size')
            self.batch_size_limit = config.get('batch_size_limit')
            self.batch_expansion_rate = config.get('batch_expansion_rate')
            self.batch_expansion_th = config.get('batch_expansion_th')
            self.batch_split_size = config.get('batch_split_size')
            self.val_batch_size = config.get('val_batch_size')

            self.optimizer = config.get('optimizer')
            self.initial_lr = config.get('initial_lr')
            self.scheduler = config.get('scheduler')
            self.aug_mode = config.get('aug_mode')
            self.weight_decay = config.get('weight_decay')
            self.loss = config.get('loss')
            self.margin = config.get('margin')
            self.tau1 = config.get('tau1')
            self.positives_per_query = config.get('positives_per_query')
            self.similarity = config.get('similarity')
            self.normalize_embeddings = config.get('normalize_embeddings')

            self.protocol = config.get('protocol')

            if self.protocol == 'baseline':
                self.epochs = config.get('baseline').get('epochs')
                self.scheduler_milestones = config.get('baseline').get('scheduler_milestones')
                self.train_file = config.get('baseline').get('train_file')
                self.val_file = config.get('baseline').get('val_file')
            elif self.protocol == 'refined':
                self.epochs = config.get('refined').get('epochs')
                self.scheduler_milestones = config.get('refined').get('scheduler_milestones')
                self.train_file = config.get('refined').get('train_file')
                self.val_file = config.get('refined').get('val_file')          

            self.print_model_info = config.get('print').get('model_info')
            self.print_model_parameters = config.get('print').get('number_of_parameters')
            self.debug = config.get('print').get('debug')
            self.weights_path = config.get('evaluate').get('weights_path')

PARAMS = Config()


