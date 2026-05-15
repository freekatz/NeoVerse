wan_series = [
    {
        "model_hash": "5ec04e02b42d2580483ad69f4e76346a",
        "model_name": "wan_video_dit",
        "model_class": "wan.modules.model.WanModel",
        "extra_kwargs": {
            "model_type": "t2v",
            "patch_size": [1, 2, 2],
            "in_dim": 16,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
        },
        "state_dict_converter": "neoverse.loaders.converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "neoverse.models.wan_video_text_encoder.WanTextEncoder",
    },
    {
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "wan.modules.vae_neoverse.WanVideoVAE",
        "state_dict_converter": "neoverse.loaders.converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        "model_hash": "9269f8db9040a9d860eaca435be61814",
        "model_name": "wan_video_dit",
        "model_class": "wan.modules.model.WanModel",
        "extra_kwargs": {
            "model_type": "t2v",
            "patch_size": [1, 2, 2],
            "in_dim": 16,
            "dim": 1536,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 12,
            "num_layers": 30,
            "eps": 1e-06,
        },
    },
]

reconstructor_series = [
    {
        "model_hash": "1a1d001a35f78f3a7796a1e719ead340",
        "model_name": "reconstructor",
        "model_class": "neoverse.auxiliary_models.WorldMirror",
    },
    {
        "model_hash": "252f1c3923a62665aee9b32f1b18afb5",
        "model_name": "reconstructor",
        "model_class": "neoverse.auxiliary_models.DepthAnything3Reconstructor",
    },
]

MODEL_CONFIGS = wan_series + reconstructor_series
VRAM_MANAGEMENT_MODULE_MAPS = {}
VERSION_CHECKER_MAPS = {}
