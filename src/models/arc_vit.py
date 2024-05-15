import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTPreTrainedModel, ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings,
    ViTEncoder,
)


class ArcViTEmbeddings(ViTEmbeddings):

    def __init__(self, config: ViTConfig):
        super().__init__(config, use_mask_token=False)

        self.label_tokens = nn.Embedding(config.num_labels, config.hidden_size)


    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if labels is not None:
            cls_tokens = cls_tokens + self.label_tokens(labels).unsqueeze(1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ArcViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = ArcViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(
        self,
        pixel_values,
        labels = None,
        interpolate_pos_encoding = False,
    ):

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, labels=labels, interpolate_pos_encoding=interpolate_pos_encoding
        )

        hidden_states = self.encoder(
            embedding_output
        )[0]

        cls_state = hidden_states[:, 0]
        return self.layernorm(cls_state)
    

class ArcVitForImageClassification(ViTPreTrainedModel):

    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.config = config

        self.num_labels = config.num_labels
        self.vit = ArcViTModel(config)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

        # init class embs to zero
        with torch.no_grad():
            self.vit.embeddings.label_tokens.weight.zero_()


    def forward(
        self,
        pixel_values,
        true_labels,
        debug=False
    ):
        bs = pixel_values.shape[0]

        # get logits for all versions
        hidden_states = self.vit(pixel_values)
        class_logits = self.classifier(hidden_states)

        true_states = self.vit(pixel_values, labels=true_labels)
        true_logits = self.classifier(true_states)

        fake_labels = torch.distributions.Categorical(logits=class_logits).sample()
        fake_states = self.vit(pixel_values, labels=fake_labels)
        fake_logits = self.classifier(fake_states)

        # get arc outputs
        ar = torch.arange(bs, device=class_logits.device, dtype=torch.long)
        tmp_class_logits = class_logits.view(bs, -1)
        tmp_fake_logits = fake_logits.view(bs, -1)
        tmp_true_logits = true_logits.view(bs, -1)

        true_arc = tmp_true_logits[ar, true_labels] - tmp_class_logits[ar, true_labels].detach()
        fake_arc = tmp_fake_logits[ar, fake_labels] - tmp_class_logits[ar, fake_labels].detach()

        # normalize class logits
        if not debug:
            class_logits = F.log_softmax(class_logits, dim=-1)

        return (
            class_logits,
            true_arc,
            fake_arc,
        )
