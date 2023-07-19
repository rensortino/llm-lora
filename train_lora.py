import fire
from utils.training import train


target_modules_mapping = {
    "falcon": [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
    "llama": [
        'q_proj', 
        'k_proj', 
        'v_proj', 
        'o_proj'
        ],
    "flan": [
        'q_proj', 
        'k_proj', 
        'v_proj', 
        'o_proj'
        ],
    "bart": [],
    "bert": [],
    "bloom": [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
        ],
    "opt": [
        'q_proj', 
        'k_proj', 
        'v_proj', 
        'o_proj'
        ],
    "mpt": [
        'Wqkv_proj', 
        'out_proj', 
        ],
}

supported_model_ids = [
    "decapoda-research/llama-7b-hf",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "bigscience/bloom-7b1",
    "tiiuae/falcon-7b",
    "google/flan-ul2",
    "mosaicml/mpt-7b",
]



if __name__ == "__main__":
    fire.Fire(train)
