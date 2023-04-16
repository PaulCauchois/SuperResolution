import os
from dotenv import load_dotenv
from pathlib import Path


dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path)

if __name__ == '__main__':

    if os.getenv('DOWNSCALE') == 'true':
        print('Downscaling initialization...')
        os.system('python Downscaling.py')
        print('Downscaling terminate...')

    if os.getenv('DATA_AUGMENTATION') == 'true':
        print('Data augmentation initialization...')
        os.system('python data_augmentation.py')
        print('Data augmentation terminate...')

    if os.getenv('UPSCALE'):
        print('Model initialization...')
        os.system('python Nina_SR_model.py')
        print('Model terminate...')

    if os.getenv('VISUALISATION') == 'true' :
        print('Starting visualization...')
        os.system('streamlit run web_page.py')
