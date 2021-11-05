from pytorch_lightning.callbacks import RichProgressBar


class SarcasmProgressBar(RichProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop('loss', None)
        return items
