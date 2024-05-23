from crslab.download import DownloadableFile

resources = {
    'nltk': {
        'version': '0.01',
        'file': DownloadableFile(
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/ERQp7_r_JuVKvEJC9jX3nwwBMKakfYsauEbsg1RzNgm0Dw?download=1',
            'ccd_nltk.zip',
            'f1c395b21e3c2a389ec71f6cf8977716585aa4e8684a2d9c8f5487f7da233cff',
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0
        },
    }
}
