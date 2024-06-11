from crslab.download import DownloadableFile

resources = {
    "none": {
        "version": "0.1",
        "file": DownloadableFile(
            "https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/Efu9s2gaIY9CkyM6GrJ-QjUBE5WQjBxvJn8UxUMqhLpbMg?download=1",
            "ccd_none.zip",
            "c64d1fd818a1f515bb9e22aceace4bd2e7cc017d972fef41c83b65b6bcc3fc46",
        ),
        "special_token_idx": {
            "pad": 0,
            "start": 1,
            "end": 2,
            "unk": 3,
            "pad_entity": 0,
            "pad_word": 0,
        },
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
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/EUGe4xpHGt5CmK_td2zLuacBcPDvCzMRGTin3nFOe7cxRg?download=1',
            'ccd_bert.zip',
            'e6efd9b0961e7526a85639fd4ff38fefc0c2f7588d9f3aa0065b9a404b1fa68e'
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
            'https://studntnu-my.sharepoint.com/:u:/g/personal/eirsteir_ntnu_no/EY7rhBuAXfxJknFtujgsm3MBMn4qKGH8R6EBcFPg0fgZAg?download=1',
            'ccd_gpt2.zip',
            '7a233f31cf07421e6411ad3d79e05d19b797e37f119105c5f09aeb1d96abafa3'
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
