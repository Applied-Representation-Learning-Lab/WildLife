class NewLinearClassifier(LinearClassifier):
    def __init__(self, *args, **kwargs):
        self.weight_decay = kwargs.pop("weight_decay", 0.0)
        self.lr = kwargs.pop("lr", 0.01)
        self.layers = kwargs.pop("layers", False)
        super().__init__(*args, **kwargs)

        if self.mlp:
            self.classification_head = torch.nn.Sequential(
                torch.nn.Linear(self.feature_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, self.num_classes)
            )

        self.train_preds = []
        self.train_targets = []

        self.val_preds = []
        self.val_targets = []

    def forward(self, images):
        features = self.model(images).flatten(start_dim=1)
        output = self.classification_head(features)

        return output

    def shared_step(self, batch, batch_idx):
        images, targets = batch[0], batch[1]

        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
    
        return loss, topk, predictions, targets

    def training_step(self, batch, batch_idx):
        loss, topk, predictions, targets = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)

        self.train_preds.extend(predictions.detach().cpu())
        self.train_targets.extend(targets.detach().cpu())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, topk, predictions, targets = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)

        self.val_preds.extend(predictions.detach().cpu())
        self.val_targets.extend(targets.detach().cpu())

        return loss
    
    def on_train_epoch_end(self):
        self.shared_confusion_matrix(step="train")

    def on_validation_epoch_end(self):
        self.shared_confusion_matrix(step="val")

    def shared_confusion_matrix(self, step):
        
        if step == "train":
            y_true = torch.stack(self.train_targets).numpy()
            preds = torch.stack(self.train_preds).numpy()
        elif step == "val":
            y_true = torch.stack(self.val_targets).numpy()
            preds = torch.stack(self.val_preds).numpy()

        # Log confusion matrix to W&B
        self.logger.experiment.log({
            f"{step}_confusion_matrix": wandb.plot.confusion_matrix(
                probs=preds,
                y_true=y_true,
                preds=None,
                class_names=self.class_names
            )
        })


        # Clear lists for next epoch
        if step == "train":
            self.train_targets = []
            self.train_preds = []
        elif step == "val":
            self.val_targets = []
            self.val_preds = []


    def get_effective_lr(self) -> float:
        """Compute the effective learning rate based on batch size and world size."""
        return self.lr * self.batch_size_per_device * self.trainer.world_size / 256

    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        if not self.freeze_model:
            parameters += self.model.parameters()
        optimizer = AdamW(
            parameters,
            lr=self.get_effective_lr(),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

def get_model(name):
    if name=="BioClip":
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        backbone = model.visual
        feature_dim = 512
    elif name=="MegaClassifier":
        backbone = EfficientNet.from_name('efficientnet-b3', num_classes=169)
        checkpoints = torch.jit.load("checkpoints/megaclassifier.pt")
        backbone.load_state_dict(checkpoints.state_dict(), strict=True)
        backbone._swish = torch.nn.Identity()
        backbone._fc = torch.nn.Identity()
        backbone._dropout = torch.nn.Identity()
        feature_dim = 1536
    elif name=="SpeciesNet":
        backbone = onnx.load("checkpoints/SpeciesNet.onnx")
        backbone = ConvertModel(backbone, experimental=True)
        backbone.MatMul_dense = torch.nn.Identity()
        feature_dim = 1280
    elif name=="ViT":
        backbone = torch.hub.load("facebookresearch/swag", model="vit_b16")
        backbone.head = torch.nn.Identity()
        feature_dim = 768
    elif name=="ViT-Unlocked":
        backbone = torch.hub.load("facebookresearch/swag", model="vit_b16", pretrained=False)
        backbone.head = torch.nn.Identity()
        feature_dim = 768

    return backbone, feature_dim

def cm_analysis(y_true, y_pred, labels, classes, name, ymap=None, figsize=(10,8)):
        """
        Generate matrix plot of confusion matrix with pretty annotations.
        The plot image is saved to disk.
        args: 
        y_true:    true label of the data, with shape (nsamples,)
        y_pred:    prediction of the data, with shape (nsamples,)
        filename:  filename of figure file to save
        labels:    string array, name the order of class labels in the confusion matrix.
                    use `clf.classes_` if using scikit-learn models.
                    with shape (nclass,).
        classes:   aliases for the labels. String array to be shown in the cm plot.
        ymap:      dict: any -> string, length == nclass.
                    if not None, map the labels & ys to more understandable strings.
                    Caution: original y_true, y_pred and labels must align.
        figsize:   the size of the figure plotted.
        """
        sns.set_theme(font_scale=1.0)

        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
            labels = [ymap[yi] for yi in labels]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
                else:
                    annot[i, j] = '%.2f%%\n%d' % (p, c)
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm = cm * 100
        cm.index.name = 'True Label'
        cm.columns.name = 'Predicted Label'
        fig, ax = plt.subplots(figsize=figsize)
        plt.yticks(va='center')

        sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Blues")
        
        plt.savefig(f"results/{name}.jpg", dpi=300, bbox_inches='tight')


