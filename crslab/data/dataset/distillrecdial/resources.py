from crslab.download import DownloadableFile

resources = {
    'bert': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://drive.usercontent.google.com/download?id=1-s2cQ4pnHkh_zKaVIHSRqm0KCsE50yPi&export=download&authuser=0&confirm=yes',
            'distillrecdial_bert.zip',
            'FB723B20CC9DAA3B8C70025DF44EB17BAA2BA7C13AC02EB9A95C07765601D793'
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 101,
            'end': 102,
            'unk': 100,
            'sent_split': 2,
            'word_split': 3,
            'pad_entity': 0,
            'pad_word': 0,
        },
    },
    'gpt2': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://drive.usercontent.google.com/download?id=1JYP8y-V6iuB3CsyzD1Hp0oFFuIgAMhBW&export=download&authuser=0&confirm=yes',
            'distillrecdial_gpt2.zip',
            '4A2C4C3924B2CEBE03BF197176E221D50BDC573BF3C485F6D024D18AD62EBD4C'
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'sent_split': 4,
            'word_split': 5,
            'pad_entity': 0,
            'pad_word': 0
        },
    }
}
