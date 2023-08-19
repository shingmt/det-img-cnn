import os
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import load_model, Model
from utils.utils import log
# from main.data_helpers import load_data_x


class CNN_Img_Module:

    #? model config
    _model = None
    _model_path = ''


    def __init__(self, config):
        if config is None or 'model_path' not in config:
            log('[!][CNN_Img_Module] no `model_path` defined', 'warning')
            return

        log('[ ][CNN_Img_Module] config', config)
        self.change_config(config)

        log('[ ][CNN_Img_Module] model_path', self._model_path)

        # """ Load model """
        # if os.path.isfile(self._model_path):
        #     self._model = load_model(self._model_path)
        # else: #? model_path not exist
        #     log('[!][CNN_Img_Module] `model_path` not exist', 'warning')
        # self._model.summary()

        return
    
    
    def change_config(self, config):
        if config is None:
            return

        #? if model_path is passed in config, load new model
        if 'model_path' in config and config['model_path'] != self._model_path:
            self._model_path = config['model_path']

            if os.path.isfile(self._model_path): #? model_path exist
                self._model = load_model(self._model_path)
            else: #? model_path not exist
                log('[!][CNN_Img_Module][change_config] `model_path` not exist', 'warning')
                self._model = None

        return


    def from_files(self, _map_ohash_inputs, callback):
        print('[ ][CNN_Img_Module][from_files] _map_ohash_inputs', _map_ohash_inputs)

        if self._model is None:
            log('[!][CNN_Img_Module][change_config] `model` not found', 'error')
            #? return empty result for each item
            result = {ohash: '' for ohash in _map_ohash_inputs.keys()}
            callback(result)
            return

        #? read rgb image
        result = {}
        note = {}
        # rgb_datas = []
        for ohash,item in _map_ohash_inputs.items(): #? for each output corresponding to each hash
            rgb_path = item[0] #? prv module is bin2img, output an array of 2 elements: [img_path, bytecode_path]
            log(f'   [ ][CNN_Img_Module][from_files] item = {item}')
            log(f'   [ ][CNN_Img_Module][from_files] rgb_path = {rgb_path}')
            if not os.path.isfile(rgb_path):
                continue
            rgb_data = Image.open(rgb_path)
            # rgb_datas.append(rgb_data)

            """ Infer """
            # print('[ ][CNN_Img_Module][from_files] rgb_data', rgb_data)
            X = np.array(rgb_data)
            print('[ ][CNN_Img_Module][from_files] X', X.shape, X.shape[0], X.shape[1], X.shape[2])
            X = np.reshape(X, (1,X.shape[0],X.shape[1],X.shape[2]))
            print('[ ][CNN_Img_Module][from_files] X reshaped', X.shape)
            preds = self._model.predict(X)
            print('[+][CNN_Img_Module][from_files] preds', preds)
            lbl_preds = preds.argmax(axis=1)
            print('[+][CNN_Img_Module][from_files] lbl_preds', lbl_preds)
            # return lbl_preds, preds

            #? Callbacks on finish
            k = 0
            result[ohash] = bool(int(lbl_preds[k]))
            note[ohash] = [float(v) for v in list(preds[k])]
            k += 1

        #! Call __onFinishInfer__ when the analysis is done. This can be called from anywhere in your code. In case you need synchronous processing
        callback(result, note)

        return