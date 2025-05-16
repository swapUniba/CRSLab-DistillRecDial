from crslab.download import DownloadableFile

resources = {
    'bert': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://drive.usercontent.google.com/download?id=1veZT9n7oOuwQcSwN3ukPpR9KArUzRR2l&export=download&authuser=0&confirm=yes',
            'llmsindataset_bert.zip',
            '3DAF68EA3747AB39CC5BF3AC3202DED3151EE9DDEEF3F2512FDE1574B47739E9'
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
            'https://drive.usercontent.google.com/download?id=14EB0vXh8N3dL8cgKbcR8jvqtu7QzRdIo&export=download&authuser=0&confirm=yes',
            'llmsindataset_gpt2.zip',
            '5A1B9464C07DE75A75B7632DA5F34C0F45F07382B688C65C8CB0065A21A80DBD'
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
