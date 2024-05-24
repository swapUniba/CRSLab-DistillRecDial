from crslab.download import DownloadableFile

resources = {
    "none": {
        "version": "0.1",
        "file": DownloadableFile(
            "https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/Efu9s2gaIY9CkyM6GrJ-QjUBE5WQjBxvJn8UxUMqhLpbMg?download=1",
            "ccd_none.zip",
            "c64d1fd818a1f515bb9e22aceace4bd2e7cc017d972fef41c83b65b6bcc3fc46",
        ),
    },
    'nltk': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/EQGwDWpHjbhDj0CaEwzjaHoBw0j6WxYbI8Pxq6U5IEZRRg?download=1',
            'ccd_nltk.zip',
            'b08117da91cc686a6450088d5f0508f3e8027a4bdd28d9a93493cfcdf5e5dd7f',
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
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/EX409RNUjwJCv7dSHFCQeIkBBTSBPcyrIZz_5W2173nZIA?download=1',
            'ccd_bert.zip',
            '029c3b08b0108905f55d8d1705bc91c5e365106da17d78932a5f277b056e079b'
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
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/Eb9AM2f5qBVKhV2gQV3sHccBgEU7afCymeOMX4tALLEP4A?download=1',
            'ccd_gpt2.zip',
            '45b2002f875bc69b98e9dbdfddddc8e5572ec6e5ee584e633e8053e47f4a271d'
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
