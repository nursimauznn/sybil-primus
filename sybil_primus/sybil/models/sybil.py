import torch
import torch.nn as nn
from sybil.models.cumulative_probability_layer import Cumulative_Probability_Layer
from sybil.models.pooling_layer import MultiAttentionPool
from sybil.models.primus_encoder import FrozenPrimusEncoder, FeatureAdapter, PRIMUS_DIM
from sybil.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer


class SybilNet(nn.Module):
    def __init__(self, args):
        super(SybilNet, self).__init__()

        self.hidden_dim = 512
        self.backbone = getattr(args, "backbone", "resnet")

        if self.backbone == "resnet":
            import torchvision

            encoder = torchvision.models.video.r3d_18(pretrained=True)
            self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        else:
            # Frozen Primus (NV-Reason-CT / COLIPRI) + trainable 864 -> 512 adapter.
            # Without weights the model only accepts precomputed Primus features.
            weights = getattr(args, "primus_weights", None)
            self.image_encoder = FrozenPrimusEncoder(weights, self.backbone) if weights else None
            self.feature_adapter = FeatureAdapter(PRIMUS_DIM, self.hidden_dim)

        self.pool = MultiAttentionPool()

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=args.dropout)

        self.prob_of_failure_layer = Cumulative_Probability_Layer(
            self.hidden_dim, args, max_followup=args.max_followup
        )

    def forward(self, x, batch=None):
        output = {}
        x = self.encode(x)
        pool_output = self.aggregate_and_classify(x)
        output["activ"] = x
        output.update(pool_output)
        output["prob"] = pool_output["logit"].sigmoid()

        return output

    def encode(self, x):
        if self.backbone == "resnet":
            return self.image_encoder(x)
        # x is either a CT volume (B, 1, 192, 192, 192) or precomputed Primus features (B, 864, 24, 24, 24)
        if x.shape[1] != PRIMUS_DIM:
            if self.image_encoder is None:
                raise ValueError("Model was built without primus_weights; pass precomputed (B, 864, 24, 24, 24) features")
            x = self.image_encoder(x)
        return self.feature_adapter(x)

    def aggregate_and_classify(self, x):
        pool_output = self.pool(x)

        pool_output["hidden"] = self.relu(pool_output["hidden"])
        pool_output["hidden"] = self.dropout(pool_output["hidden"])
        pool_output["logit"] = self.prob_of_failure_layer(pool_output["hidden"])

        return pool_output

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location="cpu")
        args = checkpoint["args"]
        model = SybilNet(args)

        # Remove 'model' from param names
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)  # type: ignore
        return model


class RiskFactorPredictor(SybilNet):
    def __init__(self, args):
        super(RiskFactorPredictor, self).__init__(args)

        self.length_risk_factor_vector = NLSTRiskFactorVectorizer(args).vector_length
        for key in args.risk_factor_keys:
            num_key_features = args.risk_factor_key_to_num_class[key]
            key_fc = nn.Linear(args.hidden_dim, num_key_features)
            self.add_module("{}_fc".format(key), key_fc)

    def forward(self, x, batch):
        output = {}
        x = self.encode(x)
        output = self.pool(x, batch)

        hidden = output["hidden"]
        for indx, key in enumerate(self.args.risk_factor_keys):
            output["{}_logit".format(key)] = self._modules["{}_fc".format(key)](hidden)

        return output

    def get_loss_functions(self):
        return ["risk_factor_loss"]
