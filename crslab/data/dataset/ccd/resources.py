from crslab.download import DownloadableFile

resources = {
    'nltk': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/Ee4P8qsnL5FOhwALDlLAsV8BvRHyBHSuU1yXE-QLhBkEDA?download=1',
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
    },
    'bert': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/EZ6U5LGfUOxDlUptY-rTVU0BUXlQMw8kRue0egYPPYVRIQ?download=1',
            'ccd_bert.zip',
            '74c1dea4f7f7beb12f9b655d95694cbe52b1464cf882639e5c8b4a3074c7df9e'
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
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/EVWiiK4FC4pBkgV4GgWoaXIBaOKwQfmfdYIrdSeI32z2Yw?download=1',
            'ccd_gpt2.zip',
            '5902885c730acb0c5aad749e3c89432582ae0f163b8caca1fdd680c7e9c8e4d3'
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
