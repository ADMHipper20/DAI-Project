import json
import os
import typing as t

import datasets

_CITATION = """\
@misc{gosling2023pippa,
      title={PIPPA: A Partially Synthetic Conversational Dataset}, 
      author={Tear Gosling and Alpin Dale and Yinhe Zheng},
      year={2023},
      eprint={2308.05884},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
Personal Interaction Pairs between People and AI (PIPPA) is a partially synthetic, community contributed and open-source conversational and roleplaying dataset generated from a subset of submitted logs to the Pygmalion project.
"""

_HOMEPAGE = "https://huggingface.co/datasets/PygmalionAI/PIPPA"

_LICENSE = "Apache 2.0"

_URL = "https://huggingface.co/datasets/PygmalionAI/PIPPA/resolve/main/"

class PIPPA(datasets.GeneratorBasedBuilder):
    """PIPPA: Personal Interaction Pairs between People and AI"""
    VERSION = datasets.Version("1.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="pippa", version=VERSION, description="The full PIPPA dataset as submitted."),
        datasets.BuilderConfig(name="pippa_deduped", version=VERSION, description="A deduped and cleaned version of PIPPA."),
        datasets.BuilderConfig(name="pippa_metharme", version=VERSION, description="Deduped PIPPA represented in the Metharme format."),
    ]

    DEFAULT_CONFIG_NAME = "pippa_deduped"

    def _info(self) -> datasets.DatasetInfo:
        # Userscript format
        if self.config.name in ["pippa", "pippa_deduped"]:
            features = datasets.Features({
                "submission_timestamp": datasets.Value("timestamp[ms]"),
                "categories": datasets.features.Sequence(datasets.Value("string")),
                "bot_id": datasets.Value("string"),
                "bot_name": datasets.Value("string"),
                "bot_greeting": datasets.Value("string"),
                "bot_definitions": datasets.Value("string"),
                "bot_description": datasets.Value("string"),
                "conversation": datasets.features.Sequence({
                    "message": datasets.Value("string"),
                    "is_human": datasets.Value("bool")
                })
            })
        # Metharme format
        else:
            features = datasets.Features({
                "prompt": datasets.Value("string"),
                "generation": datasets.Value("string")
            })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )
    
    def _split_generators(self, dl_manager: datasets.download.DownloadManager) -> t.List[datasets.SplitGenerator]:
        datafile = dl_manager.download(_URL + f"{self.config.name}.jsonl")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": datafile,
                    "split": "train"
                }
            )
        ]
    
    # I'm actually not sure what type split is
    def _generate_examples(self, filepath: str, split: t.Any) -> t.Tuple[int, dict]:
        def default(val: t.Any, d: t.Any) -> t.Any:
            return val if val is not None else d
        
        with open(filepath, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                entry = json.loads(row)
                # Userscript format
                if self.config.name in ["pippa", "pippa_deduped"]:
                    # The Features class of PIPPA does not expect anything to be null,
                    # so we convert nulls in the dataset to empty strings/lists
                    categories = default(entry["categories"], [])
                    bot_defs = default(entry["bot_definitions"], "")
                    bot_desc = default(entry["bot_description"], "")
                    yield idx, {
                        "submission_timestamp": entry["submission_timestamp"],
                        "categories": categories,
                        "bot_id": entry["bot_id"],
                        "bot_name": entry["bot_name"],
                        "bot_greeting": entry["bot_greeting"],
                        "bot_definitions": bot_defs,
                        "bot_description": bot_desc,
                        "conversation": entry["conversation"]
                    }
                # Metharme format
                else:
                    yield idx, {
                        "prompt": entry["prompt"],
                        "generation": entry["generation"]
                    }
